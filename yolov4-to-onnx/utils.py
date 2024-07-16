import json
import re

import numpy as np
import onnx
import sclblonnx as so
from onnxsim import simplify


def add_pre_post_processing_to_onnx(onnx_path: str, output_onnx_path: str):
    base_graph = so.graph_from_file(onnx_path)

    input_name = base_graph.input[0].name

    # get input shape
    input_shape = base_graph.input[0].type.tensor_type.shape.dim
    input_shape = [d.dim_value for d in input_shape]
    if input_shape[1] <= 3:  # NCHW
        width, height = input_shape[2], input_shape[3]
    else:  # NHWC
        width, height = input_shape[1], input_shape[2]

    # cleanup useless IO
    so.delete_output(base_graph, 'confs')
    so.delete_output(base_graph, 'boxes')
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
    make_yolov4_complementary_graph(base_graph, width, height)

    # Add mask to the model
    mask_bboxes(base_graph, 'final_output', 'mask-', width, height)
    so.add_output(base_graph, 'unmasked_bboxes', 'FLOAT', dimensions=[20, 6])

    # Save the model
    so.graph_to_file(base_graph, output_onnx_path, onnx_opset_version=get_onnx_opset_version(onnx_path))


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
    bboxes_indices1_squeezed = so.node('Squeeze', inputs=['bboxes_indices1'], outputs=['bboxes_indices1_squeezed'],
                                       axes=(0,))
    new_bboxes = so.node('Gather', inputs=[bboxes_name, 'bboxes_indices1_squeezed'], outputs=['unmasked_bboxes'],
                         axis=0)

    so.add_nodes(graph, [x_coordinates, y_coordinates, x_reducemean, y_coordinate, y_clipped, x_clipped,
                         bottom_center_corner,
                         bottom_center_corner_int,
                         bboxes_mask1, bboxes_indices1,
                         bboxes_indices1_squeezed, new_bboxes])
    return graph


def make_yolov4_complementary_graph(graph, width, height):
    so.add_input(graph, name='nms_sensitivity-', dimensions=[1], data_type='FLOAT')

    # constants
    so.add_constant(graph, 'c_100', np.array([100], dtype=np.int64), 'INT64')
    so.add_constant(graph, 'c_0.35', np.array([0.65], dtype=np.float32), 'FLOAT')
    so.add_constant(graph, 'inds_nms', value=np.array(2), data_type='INT64', )
    so.add_constant(graph, 'class_nms', value=np.array(1), data_type='INT64', )
    so.add_constant(graph, '128_constant', value=np.array([width, height, width, height]), data_type='FLOAT')

    # nodes
    confs_transposed = so.node('Transpose', inputs=['confs'], outputs=['confs_transposed'],
                               perm=(0, 2, 1))  # [1, 80, N]
    boxes_squeezed = so.node('Squeeze', inputs=['boxes'], outputs=['boxes_squeezed'], axes=(2,))  # [1, N, 4]
    nms_indices = so.node('NonMaxSuppression', inputs=['boxes_squeezed',
                                                       'confs_transposed',
                                                       'c_100',
                                                       'c_0.35',
                                                       'nms_sensitivity-'],
                          outputs=['input_nms'],
                          center_point_box=1)

    # nodes
    nms_gather = so.node('Gather', inputs=['input_nms', 'inds_nms'], outputs=['nms_gather'], axis=1)
    boxes_gather = so.node('Gather', inputs=['boxes_squeezed', 'nms_gather'], outputs=['boxes_gather'], axis=1)
    confs_gather = so.node('Gather', inputs=['confs_transposed', 'nms_gather'], outputs=['confs_gather'], axis=2)
    boxes_scaled = so.node('Mul', inputs=['boxes_gather', '128_constant'], outputs=['boxes_scaled'])
    classes_int = so.node('Gather', inputs=['input_nms', 'class_nms'], outputs=['classes_int'], axis=1)
    confs_classes = so.node('ReduceMax', inputs=['confs_gather'], outputs=['confs_classes'], axes=(1,), keepdims=0)
    classes = so.node('Cast', inputs=['classes_int'], outputs=['classes'], to=1)  # 0 for float
    classes_unsqueezed = so.node('Unsqueeze', inputs=['classes'], outputs=['classes_unsqueezed'], axes=(0, 2))
    confs_unsqueezed = so.node('Unsqueeze', inputs=['confs_classes'], outputs=['confs_unsqueezed'], axes=(2,))
    output = so.node('Concat', inputs=['boxes_scaled', 'confs_unsqueezed', 'classes_unsqueezed'],
                     outputs=['outputt'], axis=2)
    final_output = so.node('Squeeze', inputs=['outputt'], outputs=['final_output'], axes=(0,))

    so.add_nodes(graph, [confs_transposed, boxes_squeezed, nms_indices, nms_gather, boxes_gather, confs_gather,
                         boxes_scaled, classes_int, confs_classes, classes, classes_unsqueezed, confs_unsqueezed,
                         output, final_output])

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

    onnx.save(model, output_onnx_path)


def update_onnx_doc_string(onnx_path: str, model_means, model_stds):
    # Update the ONNX description
    graph = so.graph_from_file(onnx_path)
    # Add the model means and standard deviations to the ONNX graph description,
    # because that's used by the toolchain to populate some settings.
    graph.doc_string = json.dumps({'means': model_means, 'vars': model_stds})
    so.graph_to_file(graph, onnx_path, onnx_opset_version=get_onnx_opset_version(onnx_path))
