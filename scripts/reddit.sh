#!/usr/bin/bash

set -e

echo "Reddit"

target_dir="${1:-./data}"

url="http://snap.stanford.edu/jodie/reddit.csv"
filename="${url##*/}"

mkdir -p "$target_dir" && cd "$target_dir"
mkdir "reddit" && cd "reddit"

echo "- Downloading..."
curl -LOs "$url"

echo "- success."