#!/usr/bin/bash

set -e

echo "T-Finance & T-Social"

target_dir="${1:-./data}"

url="https://drive.google.com/drive/folders/1DKO8edxNJ3UUWq-w2f6mLlKecr1itEUr"

mkdir -p "$target_dir" && cd "$target_dir"

echo "- Downloading data..."
gdown -q --folder "$url"

mv "dataset" "t-datasets"
cd "t-datasets"

echo "- Extracting data..."
for file in *.zip; do
    unzip -qo "$file"
done

rm -f *.zip

echo "- Formatting data..."
mkdir t-finance t-social
cd ../..
python scripts/t-finance-formatter.py
python scripts/t-social-formatter.py
rm data/t-datasets/tfinance data/t-datasets/tsocial

echo "- success."