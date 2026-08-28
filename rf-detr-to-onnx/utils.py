"""
Postprocessing for RF-DETR raw ONNX exports (see export_raw_onnx.py).

Unlike the YOLO-family pipelines in this repo, RF-DETR is NMS-free by design
(DETR-style: a fixed set of object queries, matched 1:1 to ground truth during
training -- no redundant multi-box-per-location problem to suppress). So this
does NOT reuse ultralytics-to-onnx's NonMaxSuppression-based graph surgery;
instead it builds a small sigmoid + flatten + TopK + threshold-filter subgraph
and merges it onto the raw export via onnx.compose (not sclblonnx -- RF-DETR's
raw export requires opset 17 for its transformer ops, and sclblonnx's
Squeeze/Unsqueeze helper assumes the opset<=12 attribute-based signature, not
opset>=13's axes-as-input signature; onnx.compose.merge_models sidesteps that
by treating each stage as an independent, correctly-versioned ONNX model).

Class id quirk: RF-DETR's raw `labels` output has width num_classes+1 (90+1)
and its class index space is the *sparse* COCO id space (ids 1..90 with 10
gaps -- 12, 26, 29, 30, 45, 66, 68, 69, 71, 83 -- historically unused).  Every
other model in this repo uses the *compact* 0..79 ordering (see
yolov9-to-onnx/complete_onnx.py's `classes` list).  A constant remap table
(`_coco_sparse_to_compact`) is baked into the graph to convert one to the
other; any label landing on a gap id or the extra no-object slot maps to -1
and is filtered out downstream (in addition to the score threshold).
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnx.compose


_COCO_SPARSE_GAPS = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83}


def _coco_sparse_to_compact(num_classes_raw: int) -> np.ndarray:
    """remap[raw_label] -> compact 0..79 index, or -1 if invalid (gap id or no-object slot).

    RF-DETR's class_embed head indexes classes directly by COCO category id (verified by
    reading detr.py's predict() path, ~line 2515: "Pretrained COCO models use COCO category
    IDs (1-90, with gaps) as class_ids" -- raw_label == coco_id, slot 0 is background/unused,
    NOT raw_label == coco_id - 1. The earlier version of this function assumed a -1 shift,
    which silently relabels every detection as the *next* COCO category (person -> bicycle,
    car -> motorbike, etc.) -- confirmed by cross-checking against rfdetr's own model.predict()
    on the same image, which shows correct labels the ONNX output did not.
    """
    valid_ids = [i for i in range(1, num_classes_raw + 1) if i not in _COCO_SPARSE_GAPS]
    assert len(valid_ids) == 80, f"expected 80 valid COCO ids, got {len(valid_ids)}"
    remap = np.full(num_classes_raw + 1, -1, dtype=np.int64)  # index 0 = background/unused slot
    for compact_idx, coco_id in enumerate(valid_ids):
        remap[coco_id] = compact_idx  # raw_label IS the coco_id, no shift
    return remap


def _const(name, array):
    return helper.make_tensor(
        name, {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64,
               np.dtype("bool"): TensorProto.BOOL}[array.dtype],
        array.shape, array.flatten().tolist(),
    )


def build_preprocess_model(resolution: int, opset: int) -> onnx.ModelProto:
    """image- [1,3,H,W] uint8-range float -> ImageNet-normalized `input` (RF-DETR's DINOv2 backbone
    expects (pixel/255 - mean) / std, NOT the plain /255 the YOLO-family models in this repo use)."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    nodes = [
        helper.make_node("Div", ["image-", "c_255"], ["norm1"]),
        helper.make_node("Sub", ["norm1", "c_mean"], ["norm2"]),
        helper.make_node("Div", ["norm2", "c_std"], ["input"]),
    ]
    graph = helper.make_graph(
        nodes,
        "preprocess",
        [helper.make_tensor_value_info("image-", TensorProto.FLOAT, [1, 3, resolution, resolution])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, resolution, resolution])],
        initializer=[
            _const("c_255", np.array([255.0], dtype=np.float32)),
            _const("c_mean", mean),
            _const("c_std", std),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)


def build_postprocess_model(resolution: int, num_classes_raw: int, num_queries: int,
                             num_select: int, max_det: int, opset: int) -> onnx.ModelProto:
    """dets [1,Q,4] (sigmoid cxcywh, normalized) + labels [1,Q,C+1] (raw logits)
    + nms_sensitivity- [1] -> bboxes- [N,6] xyxy(pixels),score,compact_class_id, N in [0,max_det]."""
    C = num_classes_raw + 1
    remap = _coco_sparse_to_compact(num_classes_raw)

    nodes = [
        helper.make_node("Sigmoid", ["labels"], ["probs"]),
        helper.make_node("Reshape", ["probs", "c_flat_shape"], ["probs_flat"]),  # [1, Q*C]
        helper.make_node("TopK", ["probs_flat", "c_k"], ["topk_vals", "topk_idx"], axis=1, largest=1, sorted=1),
        helper.make_node("Squeeze", ["topk_vals", "c_axis0"], ["scores_all"]),   # [K]
        helper.make_node("Squeeze", ["topk_idx", "c_axis0"], ["idx_all"]),       # [K] int64

        helper.make_node("Div", ["idx_all", "c_C_i64"], ["query_idx"]),          # [K] = idx // C
        helper.make_node("Mod", ["idx_all", "c_C_i64"], ["raw_label"]),          # [K] = idx % C
        helper.make_node("Gather", ["c_remap", "raw_label"], ["compact_label"], axis=0),  # [K]

        helper.make_node("Squeeze", ["dets", "c_axis0"], ["dets_sq"]),           # [Q,4] cxcywh norm
        helper.make_node("Gather", ["dets_sq", "query_idx"], ["boxes_cxcywh"], axis=0),   # [K,4]

        # cxcywh -> xyxy (normalized), same offset trick as ultralytics-to-onnx/utils.py
        helper.make_node("Gather", ["boxes_cxcywh", "c_0101"], ["b_0101"], axis=1),
        helper.make_node("Gather", ["boxes_cxcywh", "c_2323"], ["b_2323"], axis=1),
        helper.make_node("Mul", ["c_pm_half", "b_2323"], ["b_offset"]),
        helper.make_node("Add", ["b_0101", "b_offset"], ["boxes_xyxy_norm"]),    # [K,4] in [0,1]
        helper.make_node("Mul", ["boxes_xyxy_norm", "c_scale_wh"], ["boxes_xyxy_px"]),  # [K,4] pixels

        # keep = (score > nms_sensitivity) & (compact_label >= 0)
        helper.make_node("Greater", ["scores_all", "nms_sensitivity-"], ["score_ok"]),
        helper.make_node("GreaterOrEqual", ["compact_label", "c_0_i64"], ["label_ok"]),
        helper.make_node("And", ["score_ok", "label_ok"], ["keep_mask"]),
        helper.make_node("NonZero", ["keep_mask"], ["keep_idx_2d"]),             # [1, N]
        helper.make_node("Squeeze", ["keep_idx_2d", "c_axis0"], ["keep_idx"]),   # [N]

        # Cap at max_det. topk_idx (hence keep_idx, via NonZero preserving order) is already
        # sorted descending by score, so slicing the first max_det keeps the highest-scoring
        # ones -- matching the YOLO-family models' documented "up to max_det" output contract,
        # which this graph previously declared (output shape [max_det,6]) but never enforced.
        helper.make_node("Slice", ["keep_idx", "c_slice0", "c_max_det", "c_axis0"], ["keep_idx_capped"]),  # [min(N,max_det)]

        helper.make_node("Gather", ["boxes_xyxy_px", "keep_idx_capped"], ["f_boxes"], axis=0),   # [N,4]
        helper.make_node("Gather", ["scores_all", "keep_idx_capped"], ["f_scores_1d"], axis=0),  # [N]
        helper.make_node("Gather", ["compact_label", "keep_idx_capped"], ["f_labels_1d"], axis=0),  # [N]
        helper.make_node("Cast", ["f_labels_1d"], ["f_labels_1d_f"], to=TensorProto.FLOAT),

        helper.make_node("Unsqueeze", ["f_scores_1d", "c_axis1"], ["f_scores"]),  # [N,1]
        helper.make_node("Unsqueeze", ["f_labels_1d_f", "c_axis1"], ["f_labels"]),  # [N,1]
        helper.make_node("Concat", ["f_boxes", "f_scores", "f_labels"], ["bboxes-"], axis=1),  # [N,6]
    ]

    graph = helper.make_graph(
        nodes,
        "postprocess",
        [
            helper.make_tensor_value_info("dets", TensorProto.FLOAT, [1, num_queries, 4]),
            helper.make_tensor_value_info("labels", TensorProto.FLOAT, [1, num_queries, C]),
            helper.make_tensor_value_info("nms_sensitivity-", TensorProto.FLOAT, [1]),
        ],
        [helper.make_tensor_value_info("bboxes-", TensorProto.FLOAT, [max_det, 6])],
        initializer=[
            _const("c_flat_shape", np.array([1, num_queries * C], dtype=np.int64)),
            _const("c_k", np.array([min(num_select, num_queries * C)], dtype=np.int64)),
            _const("c_axis0", np.array([0], dtype=np.int64)),
            _const("c_axis1", np.array([1], dtype=np.int64)),
            _const("c_slice0", np.array([0], dtype=np.int64)),
            _const("c_max_det", np.array([max_det], dtype=np.int64)),
            _const("c_C_i64", np.array(C, dtype=np.int64)),
            _const("c_0_i64", np.array(0, dtype=np.int64)),
            _const("c_remap", remap),
            _const("c_0101", np.array([0, 1, 0, 1], dtype=np.int64)),
            _const("c_2323", np.array([2, 3, 2, 3], dtype=np.int64)),
            _const("c_pm_half", np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float32)),
            _const("c_scale_wh", np.array([resolution, resolution, resolution, resolution], dtype=np.float32)),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)


def add_pre_post_processing_to_onnx(raw_onnx_path: str, output_path: str, classes: list,
                                     resolution: int, num_classes_raw: int = 90,
                                     num_select: int = 300, max_det: int = 20):
    raw = onnx.load(raw_onnx_path)
    opset = raw.opset_import[0].version
    num_queries = raw.graph.output[0].type.tensor_type.shape.dim[1].dim_value

    pre = build_preprocess_model(resolution, opset)
    post = build_postprocess_model(resolution, num_classes_raw, num_queries, num_select, max_det, opset)

    merged = onnx.compose.merge_models(pre, raw, io_map=[("input", "input")])
    merged = onnx.compose.merge_models(merged, post, io_map=[("dets", "dets"), ("labels", "labels")])

    classes_str = ";".join(f"{i}:{c}" for i, c in enumerate(classes))
    new_name = f"bboxes-format:xyxysc;{classes_str}"
    for out in merged.graph.output:
        if out.name == "bboxes-":
            out.name = new_name
    for node in merged.graph.node:
        node.output[:] = [new_name if o == "bboxes-" else o for o in node.output]

    import json
    merged.graph.doc_string = json.dumps({"means": [0, 0, 0], "vars": [1, 1, 1]})

    # merge_models concatenates opset_import per merge step, leaving duplicate ("", opset) entries
    del merged.opset_import[:]
    merged.opset_import.append(helper.make_opsetid("", opset))

    onnx.checker.check_model(merged)
    onnx.save(merged, output_path)


def simplify_onnx(onnx_path: str, output_onnx_path: str):
    from onnxsim import simplify
    model, check = simplify(onnx_path)
    assert check, "Failed to simplify ONNX model"
    onnx.save(model, output_onnx_path)
