from os.path import splitext

from utils import add_pre_post_processing_to_onnx, rename_io, update_onnx_doc_string

if __name__ == '__main__':
    from sys import argv

    if len(argv) != 2:
        print("Usage: python complete-onnx.py model.onnx")
        exit(1)
    onnx_path = argv[1]
    output_onnx_path = splitext(onnx_path)[0] + "-complete.onnx"
    classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    classes_str = ';'.join([f'{i}:{c}' for i, c in enumerate(classes)])

    add_pre_post_processing_to_onnx(onnx_path, output_onnx_path)

    rename_io(output_onnx_path, output_onnx_path, **{'image': 'image-',
                                                     'unmasked_bboxes': f'bboxes-format:xyxysc;{classes_str}',
                                                     })

    update_onnx_doc_string(output_onnx_path, [0, 0, 0], [1, 1, 1])
