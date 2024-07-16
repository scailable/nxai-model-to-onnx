set -e

cd "$(dirname "$0")"

# Remove the previous files
rm -rf *.onnx
# Convert the YOLOv5 model to ONNX format
bash export-to-onnx.sh ./bee-model.pt ./model.onnx 640
# Complete the ONNX with post-processing
python complete_onnx.py
# Test the ONNX model with ONNX Runtime
python test_onnx.py