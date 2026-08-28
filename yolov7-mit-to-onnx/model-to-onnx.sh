set -e

cd "$(dirname "$0")" || exit

img_size=${1:-640}

mkdir -p models weights || true
rm -f *.onnx

git clone https://github.com/MultimediaTechLab/YOLO.git repo || true

curl -sL -o weights/v7.pt \
    "https://github.com/MultimediaTechLab/YOLO/releases/download/v1.0-alpha/v7.pt"

python export_raw_onnx.py repo weights/v7.pt "v7-$img_size-raw.onnx" "$img_size"
python complete_onnx.py
python test_onnx.py

cp *-complete.onnx models
