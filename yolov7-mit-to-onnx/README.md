# MultimediaTechLab/YOLO (YOLOv7, MIT license) to ONNX

Exports the YOLOv7 checkpoint from [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)
(the same MIT-licensed repo used by [`yolov9-mit-to-onnx`](../yolov9-mit-to-onnx)) to the same
AI-Manager-compatible ONNX format produced by [`ultralytics-to-onnx`](../ultralytics-to-onnx) and
`yolov9-mit-to-onnx`.

## How this differs from yolov9-mit-to-onnx

Same underlying problem as YOLOv9 here (this repo's own `torch.onnx.export` in
`yolo/utils/deploy_utils.py` exports the undecoded multi-scale head, unusable for downstream NMS
graph-surgery), same fix shape (wrap `model -> decode -> [1, 4+C, N]`, reuse
`utils.py`'s NMS graph surgery unchanged) — but YOLOv7 uses a **different decoder**:

- **`Anc2Box`, not `Vec2Box`.** YOLOv9 is anchor-free; YOLOv7 is classic anchor-based (3 anchor
  boxes x 3 scales, strides `[8,16,32]`, anchors from `v7.yaml`). `create_converter(model_cfg.name,
  ...)` dispatches to `Anc2Box` automatically because `v7.yaml`'s `name` field is `"v7"`.
- **4-tuple return, not 3-tuple.** `Anc2Box.__call__` returns
  `(preds_cls, None, preds_box, preds_cnf)` — an extra `preds_cnf` (objectness) slot that
  `Vec2Box` doesn't have.
- **Confidence = class prob x objectness, not just class prob.** `preds_cnf` is a separate,
  already-`sigmoid()`-applied objectness score (classic YOLO design: "is there an object here" is
  scored independently from "which class is it"). Final per-class confidence fed into the NMS
  graph is `preds_cls.sigmoid() * preds_cnf`, not `preds_cls.sigmoid()` alone. Verified by hand on
  a real detection from `pedestrians.jpg`: `cls_prob=0.9996`, `objectness=0.8937`,
  `cls_prob * objectness=0.89339` — matches the value the wrapper module produces exactly. Getting
  this wrong wouldn't break the ONNX export (shapes still type-check) but would silently miss the
  objectness gate, over-triggering on background clutter that scores confidently for *some* class
  without there actually being an object there.
- `preds_box` is still already absolute pixel-space xyxy (same as `Vec2Box`), so `utils.py`'s
  xyxy-not-cxcywh NMS graph surgery — copied verbatim from `yolov9-mit-to-onnx`, no changes needed
  — applies as-is.

## Checkpoint status

Only `v7.pt` is available from this repo's model zoo — the other YOLOv7 variants (X/W6/E6/D6/E6E)
listed in its docs have no real download link. `v7.pt` is marked unverified upstream (unlike
`v9-s`/`v9-m`/`v9-c`), but it loads and runs correctly: tested on `pedestrians.jpg`, produces
plausible detections (person-shaped narrow-tall boxes at the expected y~495-580 row, a wide-squat
car detection that lines up with what `rf-detr-to-onnx`'s models independently found in the same
region of the same image) — no evidence of a broken checkpoint.

## Getting started

```bash
pip install -r requirements.txt
bash model-to-onnx.sh   # optional arg: image size, defaults to 640
```

## Beyond this example

To use a different class subset, edit `classes` in `complete_onnx.py`. `X7`/`W6`/`E6`/`D6`/`E6E`
checkpoints aren't available from this repo's releases — if you obtain one another way, the same
`export_raw_onnx.py` should apply unchanged as long as its config also dispatches to `Anc2Box`.
