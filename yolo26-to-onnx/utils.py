import json
import re

import numpy as np
import onnx
import sclblonnx as so
from onnx import save
from onnxsim import simplify
from typing import List

def add_pre_post_processing_to_onnx(onnx_path: str, output_onnx_path: str, classes: List[str]):
    base_graph = so.graph_from_file(onnx_path)
    output_name = base_graph.output[0].name
    input_name = base_graph.input[0].name

    # get input shape
    input_shape = base_graph.input[0].type.tensor_type.shape.dim
    input_shape = [d.dim_value for d in input_shape]

    # cleanup useless IO
    so.delete_output(base_graph, output_name)
    so.delete_input(base_graph, input_name)

    # Normalize the input by dividing by 255
    so.add_constant(base_graph, 'c_255', np.array([255], dtype=np.float32), 'FLOAT')
    div = so.node('Div', inputs=['image-', 'c_255'], outputs=[input_name])
    base_graph.node.insert(0, div)
    so.add_input(base_graph, name='image-', dimensions=input_shape, data_type='FLOAT')

    # move constant nodes to the beginning of the graph
    constant_nodes = [n for n in base_graph.node if n.op_type == 'Constant']
    for n in constant_nodes:
        base_graph.node.remove(n)
        base_graph.node.insert(0, n)

    # Add NMS to the model
    make_yolov5_complementary_graph(base_graph, output_name)

    # Save the model
    so.graph_to_file(base_graph, output_onnx_path, onnx_opset_version=get_onnx_opset_version(onnx_path))

    # Rename model IO
    classes_str = ';'.join([f'{i}:{c}' for i, c in enumerate(classes)])
    rename_io(output_onnx_path, output_onnx_path, **{'image': 'image-',
                                                     'bboxes-': f'bboxes-format:xyxysc;{classes_str}'
                                                     })

    update_onnx_doc_string(output_onnx_path, [0, 0, 0], [1, 1, 1])


def make_yolov5_complementary_graph(g, output_name):
    so.add_input(g, name='nms_sensitivity-', dimensions=[1], data_type='FLOAT')

    # constants
    so.add_constant(g, name='c_axis_0', value=np.array([0]), data_type='INT64')
    so.add_constant(g, name='c_axis_1', value=np.array([1]), data_type='INT64')
    so.add_constant(g, name='c_4', value=np.array([4]), data_type='INT64')

    # output = output[0] - select first element from batch dimension [1, N, 6] -> [N, 6]
    # In opset 13+, Squeeze takes axes as an input tensor
    output = so.node('Squeeze', inputs=[output_name, 'c_axis_0'], outputs=['_output'])
    
    # scores = output[:, -1] - extract last column (scores column) [N, 6] -> [N]
    scores_extracted_2d = so.node('Gather', inputs=['_output', 'c_4'], outputs=['scores_extracted_2d'], axis=1)
    scores = so.node('Squeeze', inputs=['scores_extracted_2d', 'c_axis_1'], outputs=['scores'])
    
    # mask = scores > nms_sensitivity-
    mask = so.node('Greater', inputs=['scores', 'nms_sensitivity-'], outputs=['mask'])
    
    # new_output = output[mask] - filter output rows based on mask [N, 6] -> [K, 6]
    # NonZero returns [1, K] where K is the number of True values, first dim is number of input dims
    mask_indices = so.node('NonZero', inputs=['mask'], outputs=['mask_indices'])
    mask_indices_squeezed = so.node('Squeeze', inputs=['mask_indices', 'c_axis_0'], outputs=['mask_indices_squeezed'])
    new_output = so.node('Gather', inputs=['_output', 'mask_indices_squeezed'], outputs=['bboxes-'], axis=0)

    so.add_nodes(g,
                 [output, scores_extracted_2d, scores, mask,
                  mask_indices, mask_indices_squeezed, new_output
                  ])

    so.add_output(g, 'bboxes-', 'FLOAT', dimensions=[20, 6])

    return g


def rename_io(model_path, new_model_path=None, **io_names):
    if new_model_path is None:
        new_model_path = model_path

    g = so.graph_from_file(model_path)

    def log(old: bool = True):
        s = 'Old' if old else 'New'
        assert so.list_inputs(g)
        assert so.list_outputs(g)

    if io_names == {}:
        return

    inputs = [i.name for i in g.input]
    outputs = [i.name for i in g.output]

    for k, v in io_names.items():
        pattern = re.compile(k)
        renamed = False

        for i in inputs:
            if pattern.match(i):
                renamed = True
                so.rename_input(g, i, v)
                break

        if not renamed:
            for o in outputs:
                if pattern.match(o):
                    renamed = True
                    so.rename_output(g, o, v)
                    break

        if not renamed:
            continue

    log(False)

    so.graph_to_file(g, new_model_path, onnx_opset_version=get_onnx_opset_version(model_path))


def get_onnx_opset_version(onnx_path):
    model = onnx.load(onnx_path)
    opset_version = model.opset_import[0].version if len(model.opset_import) > 0 else 0
    return opset_version


def simplify_onnx(onnx_path: str, output_onnx_path: str):
    try:
        model, check = simplify(onnx_path)
        assert check, 'Failed to simplify ONNX model'
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        raise Exception('Failed to simplify ONNX model')

    save(model, output_onnx_path)


def update_onnx_doc_string(onnx_path: str, model_means: List[float], model_stds: List[float]):
    # Update the ONNX description
    graph = so.graph_from_file(onnx_path)
    # Add the model means and standard deviations to the ONNX graph description,
    # because that's used by the toolchain to populate some settings.
    graph.doc_string = json.dumps({'means': model_means, 'vars': model_stds})
    so.graph_to_file(graph, onnx_path, onnx_opset_version=get_onnx_opset_version(onnx_path))
