#!/usr/bin/bash
# bip weighted
set -e

echo "Epinions"

target_dir="${1:-./data}"

url="http://konect.cc/files/download.tsv.epinions-rating.tar.bz2"
filename="${url##*/}"

mkdir -p "$target_dir" && cd "$target_dir"
mkdir "epinions" && cd "epinions"

echo "- Downloading..."
curl -LOs "$url"

echo "- Extracting..."
tar -xf "$filename"

rm "$filename"

mv "epinions-rating/out.epinions-rating" "epinions-edges"
rm -rf "epinions-rating"

echo "- Formatting..."
{ 
  printf 'src,tgt,weight,edge_timestamp\n'
  sed '1,1d' "epinions-edges" | sed 's/ \+/,/g'
} > epinions-edges.csv

rm "epinions-edges"

echo "- success."