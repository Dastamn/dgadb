#!/usr/bin/bash

set -e

echo "Email-DNC"
# it says asym positive ??
target_dir="${1:-./data}"
target_filename="email-dnc.edges"

url="https://nrvis.com/download/data/dynamic/email-dnc.zip"
filename="${url##*/}"

mkdir -p "$target_dir/email-dnc" && cd "$target_dir/email-dnc"

echo "- Downloading..."
curl -LOs "$url"

echo "- Extracting..."
unzip -qo "$filename"
rm "readme.html" "$filename"

echo "- Formatting..."
{ 
  printf 'src,tgt,edge_timestamp\n'
  sed '1s/^\xEF\xBB\xBF//' "$target_filename" | sed 's/ \+/,/g'
} > email-dnc.csv

rm "$target_filename"

echo "success."
