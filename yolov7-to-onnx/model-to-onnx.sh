set -e

cd "$(dirname "$0")" || exit

mkdir models || true

rm -rf *.onnx

width=1280
height=1280
model_name=yolov7x

git clone https://github.com/WongKinYiu/yolov7 || true

cd yolov7

python export.py --weights $model_name.pt --grid --end2end --simplify --include-nms \
        --topk-all 100 --iou-thres 0.5 --conf-thres 0.35 --img-size $height $width --max-wh $height

cp $model_name.onnx ../"$model_name-$width-$height.onnx"
cd ..

python complete-onnx.py "$model_name-$width-$height.onnx"

python test-onnx.py

cp *-complete.onnx models