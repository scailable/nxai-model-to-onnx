# This script is used to convert the YOLOv3/5/8 model to ONNX format opset version 12
set -e

current_dir=$(pwd)

# Check if the user has provided the model name
if [ -z "$3" ]
then
    echo "Usage: $0 <model_name> <onnx_path> <img_size>"
    exit 1
fi

model_name=$(realpath $1) # exampples: yolov5s.pt ../custom-model.pt
onnx_path=$(realpath $2) # example: ./model.onnx
img_size=$3 # example: 640

echo "Model name: $model_name"
echo "ONNX path: $onnx_path"
echo "Image size: $img_size"

git clone https://github.com/ultralytics/yolov5.git || true
cd yolov5
#pip install -r requirements.txt  # install

python export.py --weights "$model_name" --include onnx --img "$img_size" --batch 1 --opset 12

# Move the file to the specified path
model_name_without_extension=$(basename "$model_name" .pt)
mv "$model_name_without_extension.onnx" "$onnx_path" || echo "Failed to move the file to '$onnx_path'"