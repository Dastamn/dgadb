
#!/usr/bin/bash

set -e

echo "Digg-homogeneous"

target_dir="${1:-./data}"
target_filename="ia-digg-reply.edges"

url="https://nrvis.com/download/data/dynamic/ia-digg-reply.zip"
# same as http://konect.cc/networks/munmun_digg_reply/
filename="${url##*/}"

mkdir -p "$target_dir/digg-homo" && cd "$target_dir/digg-homo"

echo "- Downloading..."
curl -LOs "$url"

echo "- Extracting..."
unzip -qo "$filename"
rm "readme.html" "$filename"

echo "- Formatting..."
{ 
  printf 'src,tgt,ones,edge_timestamp\n'
  sed '1,1d' "$target_filename" | sed 's/ \+/,/g'
} > digg-homo.csv

rm "$target_filename"

echo "success."
