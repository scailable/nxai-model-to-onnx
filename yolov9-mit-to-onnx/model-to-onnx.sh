set -e

cd "$(dirname "$0")" || exit

# Other values: v9-t v9-m v9-e (v9-t and v9-e are unverified upstream, see docs/2_model_zoo)
model_version=${1:-v9-s}
img_size=${2:-640}

mkdir -p models weights || true
rm -f *.onnx

git clone https://github.com/MultimediaTechLab/YOLO.git repo || true

curl -sL -o "weights/$model_version.pt" \
    "https://github.com/MultimediaTechLab/YOLO/releases/download/v1.0-alpha/$model_version.pt"

python export_raw_onnx.py repo "$model_version" "weights/$model_version.pt" "$model_version-$img_size-raw.onnx" "$img_size"
python complete_onnx.py
python test_onnx.py

cp *-complete.onnx models
