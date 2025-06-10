#!/usr/bin/bash

set -e

echo "BITCOIN"

target_dir="${1:-./data}"

mkdir -p "$target_dir"
cd "$target_dir"

for type in alpha otc; do
    url="https://snap.stanford.edu/data/soc-sign-bitcoin${type}.csv.gz"
    filename="${url##*/}"

    echo "- Downloading..."
    curl -LOs "$url"

    echo "- Extracting..."
    gunzip "$filename"

    mv "${filename%.*}" "bitcoin-$type.csv"
done

echo "- success."
