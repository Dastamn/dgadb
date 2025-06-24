#!/usr/bin/bash
# asym positive
set -e

echo "Enron"

target_dir="${1:-./data}"

url="http://konect.cc/files/download.tsv.enron.tar.bz2"
filename="${url##*/}"

mkdir -p "$target_dir" && cd "$target_dir"
mkdir "enron" && cd "enron"

echo "- Downloading..."
curl -LOs "$url"

echo "- Extracting..."
tar -xf "$filename"

rm "$filename"

mv "enron/out.enron" "enron-edges"
rm -rf "enron"

echo "- Formatting..."
{ 
  printf 'src,tgt,ones,edge_timestamp\n'
  sed '1,1d' "enron-edges" | sed 's/ \+/,/g'
} > enron-edges.csv

rm "enron-edges"

echo "- success."