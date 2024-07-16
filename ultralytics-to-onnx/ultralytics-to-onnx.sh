set -e

cd "$(dirname "$0")" || exit

model_name=yolov8l # exampples: yolov5su, yolov8n, yolov8s, ../custom-model.pt
img_size=640

mkdir models || true

# Remove old onnx files
rm -rf *.onnx

bash export-to-onnx.sh "$model_name" "$model_name-$img_size.onnx" "$img_size"
python complete_onnx.py
python test_onnx.py

cp *-complete.onnx models