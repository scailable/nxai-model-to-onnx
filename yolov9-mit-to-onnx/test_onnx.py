from glob import glob
from os.path import join, dirname, abspath

import numpy as np
import onnxruntime as rt
from PIL import Image

PATH = dirname(abspath(__file__))


def test_model(model_path, img_path, nms_sensitivity=0.3):
    sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_shape = sess.get_inputs()[0].shape
    if input_shape[1] <= 3:  # nchw
        channels, height, width = input_shape[1], input_shape[2], input_shape[3]
    else:  # nhwc
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

    img = Image.open(img_path).convert('RGB').resize((width, height))
    img = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None]

    bboxes = sess.run(None, {
        'image-': img,
        'nms_sensitivity-': np.array([nms_sensitivity], dtype=np.float32),
    })[0]

    return bboxes, (width, height)


if __name__ == '__main__':
    model_path = glob(join(PATH, '*-complete.onnx'))[0]
    img_path = join(PATH, 'pedestrians.jpg')
    bboxes, (width, height) = test_model(model_path, img_path)
    print(f'{model_path}: {bboxes.shape[0]} detections on a {width}x{height} input')
    for x1, y1, x2, y2, score, class_id in bboxes:
        print(f'  class={int(class_id):>2}  score={score:.2f}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})')
