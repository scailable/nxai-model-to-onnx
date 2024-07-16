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
    if input_shape[1] <= 3:  # NCHW
        width, height = input_shape[2], input_shape[3]
    else:  # NHWC
        width, height = input_shape[1], input_shape[2]

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

    # Minor post-processing
    # Remove the batch dimension
    so.add_constant(base_graph, 'c_0', np.array(0), 'INT64')
    so.add_constant(base_graph, 'c_[0]', np.array([0]), 'INT64')
    so.add_constant(base_graph, 'c_4', np.array(4), 'INT64')
    squeeze = so.node('Squeeze', inputs=[output_name, 'c_[0]'], outputs=['bboxes'])
    # keep bboxes with score > nms_sensitivity-
    so.add_input(base_graph, name='nms_sensitivity-', dimensions=[1], data_type='FLOAT')
    scores = so.node('Gather', inputs=['bboxes', 'c_4'], outputs=['scores'], axis=1)
    filter_mask = so.node('Greater', inputs=['scores', 'nms_sensitivity-'], outputs=['filter_mask'])
    bboxes_indices = so.node('NonZero', inputs=['filter_mask'], outputs=['bboxes_indices'])
    bboxes_indices2 = so.node('Gather', inputs=['bboxes_indices', 'c_0'], outputs=['bboxes_indices2'], axis=0)
    bboxes = so.node('Gather', inputs=['bboxes', 'bboxes_indices2'], outputs=['bboxes-'], axis=0)
    so.add_nodes(base_graph, [squeeze, scores, filter_mask, bboxes_indices, bboxes_indices2, bboxes])

    # Add mask to the model
    mask_bboxes(base_graph, 'bboxes-', 'mask-', width, height)
    so.add_output(base_graph, 'unmasked_bboxes', 'FLOAT', dimensions=[20, 6])

    # Save the model
    so.graph_to_file(base_graph, output_onnx_path, onnx_opset_version=get_onnx_opset_version(onnx_path))

    # Rename model IO
    classes_str = ';'.join([f'{i}:{c}' for i, c in enumerate(classes)])
    rename_io(output_onnx_path, output_onnx_path, **{'image': 'image-',
                                                     'unmasked_bboxes': f'bboxes-format:xyxysc;{classes_str}'
                                                     })

    update_onnx_doc_string(output_onnx_path, [0, 0, 0], [1, 1, 1])


def mask_bboxes(graph, bboxes_name, mask_name, w, h):
    so.add_input(graph, name=mask_name, dimensions=[h, w], data_type='BOOL')

    so.add_constant(graph, 'index_one_three', np.array([0, 2]), 'INT64')
    so.add_constant(graph, 'index_four', np.array([3, 3]), 'INT64')
    so.add_constant(graph, 'hw_clip_min', np.array(0), 'FLOAT')
    so.add_constant(graph, 'w_clip_max', np.array(w - 1), 'FLOAT')
    so.add_constant(graph, 'h_clip_max', np.array(h - 1), 'FLOAT')

    x_coordinates = so.node('Gather', inputs=[bboxes_name, 'index_one_three'], outputs=['x_coordinates'], axis=1)
    y_coordinates = so.node('Gather', inputs=[bboxes_name, 'index_four'], outputs=['y_coordinates'], axis=1)
    x_reducemean = so.node('ReduceMean', inputs=['x_coordinates'], outputs=['x_reducemean'], axes=(1,), keepdims=1)
    y_coordinate = so.node('ReduceMean', inputs=['y_coordinates'], outputs=['y_coordinate'], axes=(1,), keepdims=1)
    x_clipped = so.node('Clip', inputs=['x_reducemean', 'hw_clip_min', 'w_clip_max'], outputs=['x_clipped'])
    y_clipped = so.node('Clip', inputs=['y_coordinate', 'hw_clip_min', 'h_clip_max'], outputs=['y_clipped'])

    bottom_center_corner = so.node('Concat', inputs=['y_clipped', 'x_clipped'], outputs=['bottom_center_corner'],
                                   axis=1)
    bottom_center_corner_int = so.node('Cast', inputs=['bottom_center_corner'], outputs=['bottom_center_corner_int'],
                                       to=7)
    bboxes_mask1 = so.node('GatherND', inputs=[mask_name, 'bottom_center_corner_int'], outputs=['bboxes_mask1'])
    bboxes_indices1 = so.node('NonZero', inputs=['bboxes_mask1'], outputs=['bboxes_indices1'])
    bboxes_indices1_squeezed = so.node('Squeeze', inputs=['bboxes_indices1', 'c_[0]'], outputs=['bboxes_indices1_squeezed'])
    new_bboxes = so.node('Gather', inputs=[bboxes_name, 'bboxes_indices1_squeezed'], outputs=['unmasked_bboxes'],
                         axis=0)

    so.add_nodes(graph, [x_coordinates, y_coordinates, x_reducemean, y_coordinate, y_clipped, x_clipped,
                         bottom_center_corner,
                         bottom_center_corner_int,
                         bboxes_mask1, bboxes_indices1,
                         bboxes_indices1_squeezed, new_bboxes])
    return graph

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
