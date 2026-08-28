import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

if __name__ == '__main__':
    onnx_path, resolution = sys.argv[1], int(sys.argv[2])
    session = ort.InferenceSession(onnx_path)

    img = Image.open('pedestrians.jpg').convert('RGB').resize((resolution, resolution))
    image = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None]
    nms_sensitivity = np.array([0.4], dtype=np.float32)

    outputs = session.run(None, {'image-': image, 'nms_sensitivity-': nms_sensitivity})
    detections = outputs[0]
    print(f'{detections.shape[0]} detections at threshold 0.4')
    for row in detections[:10]:
        x1, y1, x2, y2, score, cls = row
        print(f'  class={int(cls):2d} score={score:.2f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})')
