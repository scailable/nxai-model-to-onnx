# MultimediaTechLab/YOLO (YOLOv9, MIT license) to ONNX

Exports YOLOv9 checkpoints from [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)
(an MIT-licensed reimplementation of YOLOv9/v7/YOLO-RD) to the same AI-Manager-compatible ONNX
format produced by [`ultralytics-to-onnx`](../ultralytics-to-onnx).

## The quirk this pipeline works around

Unlike `yolov9-to-onnx` (which exports from the *original* WongKinYiu/yolov9 repo, whose own
`export.py --nms` already bakes in box decoding + NMS), this repo's own ONNX export
(`yolo/utils/deploy_utils.py`) does a bare `torch.onnx.export` of the **undecoded** multi-scale
detection head — three tensors per FPN level (`class_logits`, raw DFL bins, DFL-reduced LTRB
distances), not a single ready-to-use tensor. There's no equivalent of Ultralytics'
`yolo export ... simplify=true` that gives you a clean `[1, 4+nc, N]` output.

The actual box decode (grid generation, DFL distances → xyxy pixel coordinates) lives in
`yolo.utils.bounding_box_utils.Vec2Box`, used only internally for the training/validation loop —
never wired into the model's `forward()` or the export path.

**Fix**: `export_raw_onnx.py` wraps `model -> Vec2Box -> sigmoid(scores) -> cat -> transpose` in an
`nn.Module` and traces *that* through `torch.onnx.export`, producing `[1, 4+80, N]` — same shape
Ultralytics exports, so the same NMS graph-surgery machinery applies.

One real difference survives: Vec2Box decodes straight to **xyxy** pixel coordinates, not cxcywh
like Ultralytics. `utils.py` here is `ultralytics-to-onnx/utils.py` with the final cxcywh→xyxy
conversion step removed (`make_yolov9mit_complementary_graph` vs. the original
`make_yolov5_complementary_graph`) — everything else (ArgMax/ReduceMax class collapse, per-class
box-offset trick, `NonMaxSuppression`) is unchanged. `center_point_box=0` is used with
`[x1,y1,x2,y2]`-ordered (not the ONNX spec's `[y1,x1,y2,x2]`) boxes — harmless, since IoU/NMS math
is symmetric under a consistent axis relabeling.

Also unlike `yolov9-to-onnx`, no `mask-` input is added here (the existing
`demo-models/free-models`/`demo-models/models` contract this targets only uses `image-` +
`nms_sensitivity-`) — add `mask_bboxes()` from `yolov9-to-onnx/utils.py` if you need it.

## Getting started

```bash
pip install -r requirements.txt
bash model-to-onnx.sh v9-s 640   # or v9-c, v9-m — see docs/2_model_zoo in the cloned repo for what's verified
```

This clones the repo, downloads the checkpoint from the project's own GitHub release, exports the
raw decoded ONNX, adds pre/post-processing, and runs a sanity-check inference on `pedestrians.jpg`.

Verified checkpoints (per the upstream model zoo, ✅ = validated by upstream): `v9-s` (7.1M params),
`v9-m` (20.0M), `v9-c` (25.3M). `v9-t`/`v9-e` are marked unverified upstream (🔧) — use with caution.

## Beyond this example

To use a different class subset, edit the `classes` list in `complete_onnx.py`. To use `v7.pt`
instead, note `create_converter()` dispatches to a different decoder (`Anc2Box`, anchor-based, not
`Vec2Box`) — `export_raw_onnx.py` would need a v7-specific wrapper.
