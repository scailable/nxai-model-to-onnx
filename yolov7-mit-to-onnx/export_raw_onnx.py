"""
Exports a MultimediaTechLab/YOLO (YOLOv7) checkpoint to a *raw* ONNX model with the
box-decode step (Anc2Box) baked in, producing a single [1, 4+num_classes, N] tensor
matching Ultralytics' raw export convention (boxes then per-class scores) -- same shape
convention as yolov9-mit-to-onnx, so the same NMS graph-surgery in utils.py applies as-is.

Unlike YOLOv9's Vec2Box (anchor-free), YOLOv7 uses Anc2Box (classic anchor-based decode,
3 anchors x 3 scales). Two real differences from yolov9-mit-to-onnx/export_raw_onnx.py:

1. Anc2Box.__call__ returns a 4-tuple (preds_cls, None, preds_box, preds_cnf), not Vec2Box's
   3-tuple (preds_cls, preds_anc, preds_box). preds_cnf is a separate objectness score,
   already sigmoid-applied by Anc2Box itself.
2. Final per-class confidence = sigmoid(preds_cls) * preds_cnf (objectness-gated), not just
   sigmoid(preds_cls) -- classic anchor-based YOLO confidence, verified against a real image
   by manually multiplying the two source tensors for a specific detection and confirming it
   matches the value fed into the ONNX graph (see yolov7-mit-to-onnx/README.md).

preds_box is already absolute pixel-space xyxy (same situation as Vec2Box), so utils.py's
xyxy-not-cxcywh NMS graph surgery (originally written for yolov9-mit) applies unchanged.

Usage: python export_raw_onnx.py <repo_dir> <weight_path> <output_path> [img_size]
Example: python export_raw_onnx.py ./YOLO ./weights/v7.pt v7-raw.onnx 640
"""
import sys

import torch
from torch import nn


def main():
    repo_dir, weight_path, output_path = sys.argv[1:4]
    img_size = int(sys.argv[4]) if len(sys.argv) > 4 else 640

    sys.path.insert(0, repo_dir)
    from omegaconf import OmegaConf
    from yolo.model.yolo import create_model
    from yolo.utils.bounding_box_utils import create_converter

    model_cfg = OmegaConf.load(f"{repo_dir}/yolo/config/model/v7.yaml")
    model = create_model(model_cfg, weight_path=weight_path, class_num=80).eval()
    anc2box = create_converter(model_cfg.name, model, model_cfg.anchor, [img_size, img_size], "cpu")

    class Wrapper(nn.Module):
        def __init__(self, model, anc2box):
            super().__init__()
            self.model = model
            self.anc2box = anc2box

        def forward(self, x):
            predicts = self.model(x)["Main"]
            preds_cls, _preds_anc, preds_box, preds_cnf = self.anc2box(predicts)
            scores = preds_cls.sigmoid() * preds_cnf  # class prob * objectness
            out = torch.cat([preds_box, scores], dim=-1)  # [B, N, 4+C] xyxy + scores
            return out.transpose(1, 2)  # [B, 4+C, N], matches Ultralytics' raw output layout

    wrapper = Wrapper(model, anc2box).eval()
    dummy = torch.rand(1, 3, img_size, img_size)

    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Decoded output shape: {tuple(out.shape)}")

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported raw ONNX to {output_path}")


if __name__ == "__main__":
    main()
