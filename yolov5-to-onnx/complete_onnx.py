from os.path import dirname, abspath, join, splitext

from utils import add_pre_post_processing_to_onnx, simplify_onnx

from glob import glob

HERE = dirname(abspath(__file__))

if __name__ == '__main__':
    onnx_path = glob(join(HERE, '*.onnx'))[0]
    output_onnx_path = splitext(onnx_path)[0] + '-complete.onnx'
    classes = ['bee', 'drone', 'pollenbee', 'queen']
    add_pre_post_processing_to_onnx(onnx_path, output_onnx_path, classes)
    simplify_onnx(output_onnx_path, output_onnx_path)
