# This script is used to convert the YOLOv3/5/8 model to ONNX format opset version 12
set -e

# Check if the user has provided the model name
if [ -z "$3" ]
then
    echo "Usage: $0 <model_name> <onnx_path> <img_size>"
    exit 1
fi

model_name=$1 # exampples: yolov5su, yolov8n, yolov8s, ../custom-model.pt
onnx_path=$2 # example: ../yolov8n.onnx
img_size=$3 # example: 640

# additional arguments: half, int8, optimize
yolo export model="$model_name" format=onnx imgsz="$img_size" opset=12 simplify=true

# Move the file to the specified path
mv "$model_name.onnx" "$onnx_path" || echo "Failed to move the file to '$onnx_path'"