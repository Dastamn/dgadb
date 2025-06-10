#!/usr/bin/bash

set -e

echo "MOOC"

target_dir="${1:-./data}"

url="https://snap.stanford.edu/data/act-mooc.tar.gz"
filename="${url##*/}"

mkdir -p "$target_dir" && cd "$target_dir"

echo "- Downloading..."
curl -LOs "$url"

echo "- Extracting..."
tar -xf "$filename"

rm "$filename"

echo "- success."