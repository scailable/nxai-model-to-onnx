set -e

cd "$(dirname "$0")" || exit

mkdir models || true

rm -rf *.onnx

width=640
height=640
# Other values: yolov9-s-converted.pt yolov9-m-converted.pt yolov9-c-converted.pt yolov9-e-converted.pt yolov9-s.pt yolov9-m.pt yolov9-c.pt yolov9-e.pt
model_name=yolov9-e

#pip install -r requirements.txt

git clone https://github.com/WongKinYiu/yolov9 || true

cd yolov9

#pip install -r requirements.txt

python export.py --weights "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/$model_name.pt" --include onnx --simplify --opset 12 \
        --topk-all 100 --iou-thres 0.5 --conf-thres 0.35 --imgsz $height $width --batch-size 1 --nms

cp $model_name.onnx ../$model_name-$width-$height.onnx
cd ..

python complete_onnx.py
python test_onnx.py

cp *-complete.onnx models