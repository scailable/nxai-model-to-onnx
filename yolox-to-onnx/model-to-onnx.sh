set -e

cd "$(dirname "$0")" || exit

width=640
height=640
onnx_path=yolox_s.onnx

pip install -r requirements.txt

wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
python3 YOLOX/tools/export_onnx.py --output-name $onnx_path -n yolox-s -c yolox_s.pth --opset 12 \
        --img-width $width --img-height $height

python build_npy_files.py $width $height

python complete-onnx.py $onnx_path

python test-onnx.py