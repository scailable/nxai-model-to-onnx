set -e
cd "$(dirname "$0")" || exit

variant=${1:-nano}   # nano | small
res=${2:-}           # optional override; auto-detected from the model config otherwise

pip install -q -r requirements.txt

python export_raw_onnx.py "$variant" "raw-$variant.onnx" | tee /tmp/rfdetr-export.log
resolution=${res:-$(grep -oP 'resolution=\K[0-9]+' /tmp/rfdetr-export.log | head -1)}

python complete_onnx.py "raw-$variant.onnx" "rfdetr-$variant-complete.onnx" "$resolution"
python test_onnx.py "rfdetr-$variant-complete.onnx" "$resolution"
