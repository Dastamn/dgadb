#!/usr/bin/bash

set -e

echo "Wiki"

target_dir="${1:-./data}"

url="http://snap.stanford.edu/jodie/wikipedia.csv"
filename="${url##*/}"

mkdir -p "$target_dir" && cd "$target_dir"
mkdir "wiki" && cd "wiki"

echo "- Downloading..."
curl -LOs "$url"

echo "- success."