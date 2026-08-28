import sys

from utils import add_pre_post_processing_to_onnx, simplify_onnx

CLASSES = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse', 'remote', 'keyboard',
           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']

if __name__ == '__main__':
    raw_onnx_path, output_onnx_path, resolution = sys.argv[1], sys.argv[2], int(sys.argv[3])
    add_pre_post_processing_to_onnx(raw_onnx_path, output_onnx_path, CLASSES, resolution=resolution)
    simplify_onnx(output_onnx_path, output_onnx_path)
