"""
Exports a MultimediaTechLab/YOLO (YOLOv9) checkpoint to a *raw* ONNX model with the
box-decode step (Vec2Box) baked in, producing a single [1, 4+num_classes, N] tensor
matching Ultralytics' raw export convention (boxes then per-class scores), EXCEPT the
boxes here are already xyxy pixel coordinates (Vec2Box decodes straight to xyxy),
not cxcywh like Ultralytics. See README.md for why this step is needed at all: this
repo's own `torch.onnx.export` (yolo/utils/deploy_utils.py) exports the undecoded
multi-scale head instead, which is not usable for downstream NMS graph-surgery.

Usage: python export_raw_onnx.py <repo_dir> <model_version> <weight_path> <output_path> [img_size]
Example: python export_raw_onnx.py ./YOLO v9-s ./weights/v9-s.pt v9-s-raw.onnx 640
"""
import sys

import torch
from torch import nn


def main():
    repo_dir, model_version, weight_path, output_path = sys.argv[1:5]
    img_size = int(sys.argv[5]) if len(sys.argv) > 5 else 640

    sys.path.insert(0, repo_dir)
    from omegaconf import OmegaConf
    from yolo.model.yolo import create_model
    from yolo.utils.bounding_box_utils import create_converter

    model_cfg = OmegaConf.load(f"{repo_dir}/yolo/config/model/{model_version}.yaml")
    model = create_model(model_cfg, weight_path=weight_path, class_num=80).eval()
    vec2box = create_converter(model_cfg.name, model, model_cfg.anchor, [img_size, img_size], "cpu")

    class Wrapper(nn.Module):
        def __init__(self, model, vec2box):
            super().__init__()
            self.model = model
            self.vec2box = vec2box

        def forward(self, x):
            predicts = self.model(x)["Main"]
            preds_cls, _preds_anc, preds_box = self.vec2box(predicts)
            scores = preds_cls.sigmoid()
            out = torch.cat([preds_box, scores], dim=-1)  # [B, N, 4+C] xyxy + scores
            return out.transpose(1, 2)  # [B, 4+C, N], matches Ultralytics' raw output layout

    wrapper = Wrapper(model, vec2box).eval()
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
