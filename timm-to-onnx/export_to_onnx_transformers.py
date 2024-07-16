import json
from os.path import join, dirname, abspath

import onnx
import sclblonnx as so
import timm
import torch
from torch import nn
from onnxsim import simplify

PATH = dirname(abspath(__file__))

model_name = 'vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k'
classes = ['tench', 'goldfish', 'greate white shark']  # Replace with the classes that the model was trained on.
concatenated_classes = ';'.join([f'{i}:{c}' for i, c in enumerate(classes)])
onnx_opset_version = 14
output_onnx_path = join(PATH, f'{model_name}.onnx')

input_width = 224  # Replace with your input width
input_height = 224  # Replace with your input height
model_means = [0.4850, 0.4560, 0.4060]  # Replace with your model means
model_means = [255 * m for m in model_means]  # Convert to 0-255 range
model_stds = [0.2290, 0.2240, 0.2250]  # Replace with your model standard deviations
model_stds = [255 * s for s in model_stds]  # Convert to 0-255 range

# Define the model since the model's output needs to be softmaxed
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.sotfmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.sotfmax(x)
        return x

# Load the model
model = Model()
# Set the model to evaluation mode
model.eval()
# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
print(transforms)  # NOTE: Examine the input width & height, means and stds in the transforms

# Define onnx IO names
input_names = ['image-']
output_names = [f'scores-{concatenated_classes}']
dummy_input = torch.randn(1, 3, input_width, input_height)

# Export model to ONNX
torch.onnx.export(model, dummy_input, output_onnx_path,
                  input_names=input_names, output_names=output_names,
                  opset_version=onnx_opset_version)

# Update the ONNX description
graph = so.graph_from_file(output_onnx_path)
# Add the model means and standard deviations to the ONNX graph description,
# because that's used by the toolchain to populate some settings.
graph.doc_string = json.dumps({'means': model_means, 'vars': model_stds}, default=lambda x: round(x, 2))
so.graph_to_file(graph, output_onnx_path, onnx_opset_version=onnx_opset_version)

# Simplify the ONNX model
# This step is optional, but it is recommended to reduce the size of the model
# optimize the model for inference
try:
    model = onnx.load(output_onnx_path)
    model, check = simplify(model, check_n=2)
    assert check, "Couldn't simplify the ONNX model"
    onnx.save_model(model, output_onnx_path)
except Exception as e:
    print(f'Simplification failed: {e}')
    exit(1)
