# Image classification Models

This guide shows how a **trained** image classification model (implemented using PyTorch) can be exported to ONNX the
right way.  
We'll be using image classification models implemented in the [timm](https://huggingface.co/docs/timm/quickstart)
library.

## Getting started

Make sure to install the required packages:

```bash
pip install -r requirements.txt
```

## Exporting the model to ONNX

The following command exports a trained image classification model to ONNX and save it locally:

```bash
python export_to_onnx.py
```

The exported ONNX can be uploaded on the platform and used for inference.

## Beyond this example

This example is a starting point for exporting image classification models to ONNX.   
Yet, this approach is valid for any image classification model implemented in PyTorch,
given that the model has one input (image input) and one output (representing the probabilities).

To adapt this example to your own model, you need to:

- update these variables in the Python script: `classes`, `input_width`, `input_height`, `model_means`,
  and `model_stds`.
- update the `model` variable to load your own model.


