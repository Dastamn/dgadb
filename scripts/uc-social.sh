#!/usr/bin/bash

set -e

echo "UC-SOCIAL"

target_dir="${1:-./data}"
target_filename="uc-social.csv"

url="http://konect.cc/files/download.tsv.opsahl-ucsocial.tar.bz2"
filename="${url##*/}"

mkdir -p "$target_dir" && cd "$target_dir"

echo "- Downloading..."
curl -LOs "$url"

unpacked_dir=$(tar -tf "$filename" | head -n 1 | cut -d'/' -f1)

echo "- Extracting..."
tar -xf "$filename"

mv "$unpacked_dir/out.opsahl-ucsocial" "$target_filename"

rm "$filename"
rm -r "$unpacked_dir"

echo "- success."
