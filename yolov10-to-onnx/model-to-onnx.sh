set -e

cd "$(dirname "$0")" || exit

model_name=yolov10n # yolov10{n/s/m/b/l/x}
img_size=640                  # model input size

mkdir models || true
# Remove old onnx files
rm -rf *.onnx

pip install -r requirements.txt
# Export form PyTorch to ONNX
yolo export model="$model_name" format=onnx imgsz="$img_size" opset=16 simplify=true
# Add the missing parts to the ONNX model
python complete_onnx.py
# Test the ONNX model
python test_onnx.py

# Copy model to models directory
cp *-complete.onnx models

