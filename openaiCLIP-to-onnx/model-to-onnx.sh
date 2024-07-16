set -e

cd "$(dirname "$0")" || exit

pip install -r requirements.txt

python export_to_onnx.py