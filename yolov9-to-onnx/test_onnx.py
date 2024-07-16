from os.path import join, dirname, abspath

import cv2
import numpy as np
import onnxruntime as rt

PATH = dirname(abspath(__file__))


def test_model(model_path, img_path):
    sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # get input name
    input_name1 = sess.get_inputs()[0].name
    input_name2 = sess.get_inputs()[1].name
    input_name3 = sess.get_inputs()[2].name

    # get input dimensions
    input_shape = sess.get_inputs()[0].shape
    if input_shape[1] <= 3:  # nchw
        channels, height, width = input_shape[1], input_shape[2], input_shape[3]
    else:  # nhwc
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    if channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)  # add channel dimension
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.transpose(img, (2, 0, 1)).astype('float32')
    img = np.expand_dims(img, axis=0)

    mask_area = np.repeat(1, width * height).astype('bool')
    mask_area = mask_area.reshape((height, width))
    mask_area[:, :width // 2] = 0  # mask the left part of the image

    bboxes = sess.run(None, {
        input_name1: img,
        input_name2: np.array([0.15]).astype('float32'),
        input_name3: mask_area
    })[0]


    print(bboxes.shape)

    return bboxes, (width, height)


def visualize_bboxes(bboxes, img_path, width, height):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    for bbox in bboxes:
        x1, y1, x2, y2, score, class_id = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{int(class_id)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    from glob import glob

    model_path = glob(join(PATH, '*-complete.onnx'))[0]
    img_path = join(PATH, 'pedestrians.jpg')
    bboxes, (width, height) = test_model(model_path, img_path)
    visualize_bboxes(bboxes, img_path, width, height)
