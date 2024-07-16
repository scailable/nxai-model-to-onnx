import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime

from os.path import join, split
from shutil import  move

from tool.utils import *
from tool.darknet2onnx import *


WEIGHTS = join(split(__file__)[0], 'weights', 'yolov4-tiny.weights')
SAVE_PATH = join(split(__file__)[0], '..')
CFG_FILE = join(split(__file__)[0], 'cfg', 'yolov4-tiny.cfg')

def main(IN_IMAGE_H, IN_IMAGE_W, KEEP_LARGE_OBJECTS=True, KEEP_SMALL_OBJECTS=True):
    batch_size = 1
    weight_file = WEIGHTS
    onnx_file_name = "yolov4-{}-{}.onnx".format(IN_IMAGE_H, IN_IMAGE_W)
    cfg_file = CFG_FILE

    # Transform to onnx as specified batch size
    onnx_path_demo = transform_to_onnx(cfg_file, weight_file, batch_size, width=IN_IMAGE_W, height=IN_IMAGE_H,
                                       keep_large_objects=KEEP_LARGE_OBJECTS, keep_small_objects=KEEP_SMALL_OBJECTS)

    move(onnx_path_demo, join(SAVE_PATH, onnx_file_name))

if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    if len(sys.argv) == 3:
        IN_IMAGE_H = int(sys.argv[1])
        IN_IMAGE_W = int(sys.argv[2])

        main(IN_IMAGE_H, IN_IMAGE_W)
    elif len(sys.argv) == 5:
        IN_IMAGE_H = int(sys.argv[1])
        IN_IMAGE_W = int(sys.argv[2])
        large_objects = bool(int(sys.argv[3]))
        small_objects = bool(int(sys.argv[4]))

        print("Converting to onnx and running demo ...")
        print('Keep large objects: ', large_objects)
        print('Keep small objects: ', small_objects)
        print('image width: ', IN_IMAGE_W)
        print('image height: ', IN_IMAGE_H)

        main(IN_IMAGE_H, IN_IMAGE_W, KEEP_LARGE_OBJECTS=large_objects, KEEP_SMALL_OBJECTS=small_objects)
    else:
        print('Please run this way:\n')
        print('  python demo_onnx.py <IN_IMAGE_H> <IN_IMAGE_W>')
