#!/usr/bin/bash

set -e

echo "AS-Topology"

target_dir="${1:-./data}"
target_filename="tech-as-topology.edges"

url="https://nrvis.com/download/data/dynamic/tech-as-topology.zip"
filename="${url##*/}"

mkdir -p "$target_dir/as-topology" && cd "$target_dir/as-topology"

echo "- Downloading..."
curl -LOs "$url"

echo "- Extracting..."
unzip -qo "$filename"
rm "readme.html" "$filename"

echo "- Formatting..."
{ 
  echo "src,tgt,ones,edge_timestamp"; 
  sed '1,5d' tech-as-topology.edges | sed 's/ \+/,/g'; 
} > as-topology-edges.csv

rm "tech-as-topology.edges"

echo "success."