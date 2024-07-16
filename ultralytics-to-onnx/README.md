# Ultralytics Models

This guide shows how to export a **trained** model based on **Ultralytics** to ONNX that can be directly uploaded and
deployed on servers.

For the sake of example, we'll be using the YOLOv8n.

## Getting started

Make sure to install the required packages:

```bash
pip install -r requirements.txt
```

## Exporting the model to ONNX

The ONNX model exported directly using Ultralytics CLI doesn't come with the necessary post-processing steps (namely,
Non-Maximum Suppression). Therefore, we need to add that post-processing to the ONNX model using another script.

1. So, first we export the model to ONNX using the following command:

```bash
bash export-to-onnx.sh yolov8n yolov8n.onnx 640
```

2. Then, we add the post-processing steps to the ONNX model using the following command:

```bash
 python complete_onnx.py 
```

3. Finally, to test the ONNX model, we can use the following command:

```bash
python test_onnx.py
```

The exported ONNX can be uploaded on the platform and used for inference.

## All in one

The [ultralytics-to-onnx.sh](ultralytics-to-onnx.sh) script automates the above steps. You can run it as follows:

```bash
bash ultralytics-to-onnx.sh
```

## Beyond this example

This example is a starting point for exporting Ultralytics models to ONNX.

To adapt this example to your own model, you need to:

- Call `bash export-to-onnx.sh` with the appropriate arguments 
  (in particular the first and last arguments: model name, and input size).
- Update the `complete_onnx.py` script to add the classes your model is trained on.