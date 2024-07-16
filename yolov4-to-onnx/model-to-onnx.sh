set -e

cd "$(dirname "$0")"

width=640
height=640

mkdir models || true
rm -rf *.onnx

cd pytorch-yolov4-modified

python demo_darknet2onnx.py $width $height

cd ..

python complete-onnx.py "yolov4-$width-$height.onnx"
python test-onnx.py

cp *-complete.onnx models