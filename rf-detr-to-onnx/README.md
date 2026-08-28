# RF-DETR (Apache-2.0) to ONNX

Exports RF-DETR checkpoints ([roboflow/rf-detr](https://github.com/roboflow/rf-detr),
Apache-2.0) to the same AI-Manager-compatible ONNX format produced by
[`ultralytics-to-onnx`](../ultralytics-to-onnx) and [`yolov9-mit-to-onnx`](../yolov9-mit-to-onnx).

## Why this pipeline looks different from the YOLO-family ones

RF-DETR is NMS-free by design (DETR-style: a fixed set of object queries matched 1:1 to ground
truth during training, so there's no dense-anchor redundancy to suppress). Concretely:

- **No `NonMaxSuppression` node** — the postprocessing is sigmoid → flatten → `TopK` (top 300
  query/class pairs, already score-sorted) → threshold filter (`nms_sensitivity-`) → `Slice` to
  the first `max_det` (20). **The `Slice` step was missing in the first version** — the graph
  declared a `[max_det,6]` output shape but never actually truncated to it, so raw output could
  reach ~300 detections/image at low thresholds (found while building `../../demo-models/BENCHMARKS.md`,
  where it didn't affect results since `pycocotools` truncates to its own `maxDets` regardless).
  Fixed by slicing `keep_idx` to `[0:max_det]` before the final `Gather`s — safe because `TopK`
  already sorted by score and `NonZero` preserves that order, so the first `max_det` are the
  highest-scoring survivors, not an arbitrary truncation.
- **Not `sclblonnx`** — the raw export requires **opset 17** (`LayerNormalization`, deformable
  attention, etc. aren't available at opset 12), and `sclblonnx`'s `Squeeze`/`Unsqueeze` helpers
  assume the opset-≤12 attribute-based signature, not opset-≥13's axes-as-input signature. Instead
  `utils.py` builds two small standalone ONNX models (pre/postprocess) with `onnx.helper` and
  stitches them onto RF-DETR's own official `model.export()` output via `onnx.compose.merge_models`
  — simpler and version-correct. Opset 17 is still well within what `nxai-model-to-onnx/README.md`
  documents as supported (ONNX Runtime 1.17 handles up to opset 20); it's just higher than the
  YOLO-family examples happen to use.
- **ImageNet normalization, not `/255`** — RF-DETR's DINOv2 backbone expects
  `(pixel/255 - mean) / std` with `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` (verified
  against the package's own `rfdetr/export/_tflite/inference.py` constants and by cross-checking
  against `model.predict()`'s own output). This is baked into the graph exactly like every other
  model here bakes in its own normalization.
- **Sparse → compact class remap** — RF-DETR's raw class head has width `num_classes+1` (91).
  Slot 0 is the background/no-object slot; slots 1-90 correspond **directly** to COCO category ids
  1-90 (with 10 historical gaps: 12, 26, 29, 30, 45, 66, 68, 69, 71, 83) — `raw_label == coco_id`,
  verified by reading `detr.py`'s `predict()` path (~line 2515) rather than assumed. Every other
  model in this repo uses the *compact* 0-79 ordering (see `yolov9-to-onnx/complete_onnx.py`'s
  `classes` list). `utils.py` bakes a 91-entry remap constant into the graph; a raw label landing
  on a gap id or the background slot maps to `-1` and gets filtered out alongside the score
  threshold.

  **A first version of this remap had an off-by-one bug** (`remap[coco_id - 1] = compact_idx`,
  assuming `raw_label == coco_id - 1`), which silently relabeled every detection as the *next*
  COCO category — `person`→`bicycle`, `car`→`motorbike`, `traffic light`→`fire hydrant`, etc.
  Caught by cross-checking against RF-DETR's own `model.predict()` on the same test image: its
  output was `person`/`car`/`traffic light` at scores matching the ONNX output almost exactly,
  just one category off. Fixed to `remap[coco_id] = compact_idx`; re-verified against the same
  image (`car` at (56,412,209,478) in `rfdetr-small.onnx` now scores 0.93, matching
  `model.predict()`'s 0.93 for the same box, was mislabeled `motorbike` before the fix).

## Getting started

```bash
pip install -r requirements.txt
bash model-to-onnx.sh nano   # or: bash model-to-onnx.sh small
```

This exports the raw ONNX via RF-DETR's own `model.export()`, adds pre/postprocessing, and runs a
sanity-check inference on `pedestrians.jpg`.

Only Nano (384×384) and Small (512×512) are converted — both Apache-2.0. Medium/Large are also
Apache-2.0 but out of scope here; **XL/2XL are licensed under "PML 1.0", not Apache-2.0 — do not
convert those without separately checking that license.**

## Beyond this example

To use a different class subset, edit `CLASSES` in `complete_onnx.py`. To add Medium/Large, pass
their class names to `export_raw_onnx.py`'s `variant` dispatch — the same pipeline applies as-is
since they share the same head shape convention.
