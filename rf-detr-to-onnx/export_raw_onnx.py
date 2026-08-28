"""
Exports an RF-DETR (roboflow/rf-detr, Apache-2.0) model to raw ONNX using the
package's own official exporter. Produces two outputs:
  - dets   [1, Q, 4]  sigmoid-normalized cxcywh boxes in [0,1]
  - labels [1, Q, C+1] raw (pre-sigmoid) per-class logits, C=90 COCO classes
    + 1 unused/no-object slot (class_embed is always sized num_classes+1).
No decode/NMS baked in yet -- that's added by complete_onnx.py.

Usage: python export_raw_onnx.py <variant: nano|small> <output_path>
"""
import sys


def main():
    variant, output_path = sys.argv[1], sys.argv[2]
    from rfdetr import RFDETRNano, RFDETRSmall

    cls = {"nano": RFDETRNano, "small": RFDETRSmall}[variant]
    model = cls()
    resolution = model.model_config.resolution
    num_classes = model.model_config.num_classes
    num_queries = model.model_config.num_queries
    print(f"variant={variant} resolution={resolution} num_classes={num_classes} num_queries={num_queries}")

    import os
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    out_name = os.path.basename(output_path)
    exported = model.export(output_dir=out_dir, opset_version=17, shape=(resolution, resolution))
    exported = str(exported)
    final_path = os.path.join(out_dir, out_name)
    if os.path.abspath(exported) != os.path.abspath(final_path):
        os.replace(exported, final_path)
    print(f"raw_onnx={final_path}")
    print(f"resolution={resolution}")
    print(f"num_classes={num_classes}")


if __name__ == "__main__":
    main()
