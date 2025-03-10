# Yolov12 to ONNX

This guide walks you through the process of exporting a model from Yolov12 to ONNX format. The model is then used to perform inference on a sample image.

Yolov12 models are usually available as `.pt` files. They are exported to ONNX using the `yolo` CLI.   
The generated ONNX file outputs proposed bounding boxes that need to be post-processed to get the final pertinent bounding boxes, and discard the rest. That post-processing is included the ONNX file by
running the `complete_onnx.py` script. Additionally, the script adds two inputs to the model:`nms_sensitivity-` and `mask-` 
and renames the model's inputs and outputs to match the Nx AI Manager's requirements.

## Getting started

For the sake of example, we will be using the Yolov10n model.

Now, let's convert the model to ONNX format.

### Step 1: Install Python requirements

```bash
pip install -r requirements.txt
```

### Step 2: Export the model to ONNX

```bash
yolo export model=yolov12n.pt format=onnx imgsz=640 opset=12 simplify=true
```

### Step 3: Add post-processing to the model and rename its I/O

```bash
python complete_onnx.py
```

### Step 4: Run inference on a sample image

```bash
python test_onnx.py
```

### Step 5: Deploy the model

The generated ONNX file (file with `-complete.onnx` extension) can now be uploaded in the Nx AI Cloud for deployment.


## All in one

The [model-to-onnx.sh](model-to-onnx.sh) script automates the above steps. You can run it as follows:

```bash
bash model-to-onnx.sh
```

## Beyond this example

This example is a starting point for exporting Yolov12 models to ONNX compatible with the Nx AI Manager.

To adapt this example to your own model, you need to:

- Update the trained model name/path and image size in `model-to-onnx.sh`.
- Update the `complete_onnx.py` script by changing the classes your model is trained on.