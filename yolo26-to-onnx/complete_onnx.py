from os.path import dirname, abspath, join, splitext

import onnx

from utils import add_pre_post_processing_to_onnx, simplify_onnx

from glob import glob

HERE = dirname(abspath(__file__))

if __name__ == '__main__':
    onnx_path = glob(join(HERE, '*.onnx'))[0]
    output_onnx_path = splitext(onnx_path)[0] + '-complete.onnx'
    classes = ['Person', 'Bicycle', 'Car', 'Motorbike', 'Aeroplane', 'Bus', 'Train', 'Truck', 'Boat', 'Traffic light',
               'Fire hydrant', 'Stop sign', 'Parking meter', 'Bench', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cow',
               'Elephant', 'Bear', 'Zebra', 'Giraffe', 'Backpack', 'Umbrella', 'Handbag', 'Tie', 'Suitcase', 'Frisbee',
               'Skis', 'Snowboard', 'Sports ball', 'Kite', 'Baseball bat', 'Baseball glove', 'Skateboard', 'Surfboard',
               'Tennis racket', 'Bottle', 'Wine glass', 'Cup', 'Fork', 'Knife', 'Spoon', 'Bowl', 'Banana', 'Apple',
               'Sandwich', 'Orange', 'Broccoli', 'Carrot', 'Hot dog', 'Pizza', 'Donut', 'Cake', 'Chair', 'Sofa',
               'Potted plant', 'Bed', 'Dining table', 'Toilet', 'Tv monitor', 'Laptop', 'Mouse', 'Remote', 'Keyboard',
               'Cell phone', 'Microwave', 'Oven', 'Toaster', 'Sink', 'Refrigerator', 'Book', 'Clock', 'Vase',
               'Scissors', 'Teddy bear', 'Hair drier', 'Toothbrush']
    simplify_onnx(onnx_path, output_onnx_path)
    add_pre_post_processing_to_onnx(output_onnx_path, output_onnx_path, classes)
    simplify_onnx(output_onnx_path, output_onnx_path)
