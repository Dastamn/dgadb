#!/usr/bin/bash

set -e

echo "TRACE & THEIA"

target_dir="${1:-./data}"

url="https://github.com/JakubReha/ProvCTDG/releases/download/v1.0.0.0/DATA.zip"
filename="${url##*/}"

mkdir -p "$target_dir" && cd "$target_dir"

echo "- Downloading data..."
curl -LOs "$url"

echo "- Extracting data..."
unzip -q "$filename"

rm -rf "__MACOSX"
mv "DATA" "trace-theia"

rm "$filename"

mkdir -p "trace-theia/labels"

labelfiles=("theia_browser_extension" "theia_firefox_backdoor" "trace_browser_extension" "trace_firefox_backdoor" "trace_thunderbird_phishing_exe")

echo "- Downloading labels..."

for labelfile in "${labelfiles[@]}"; do
    url="https://raw.githubusercontent.com/JakubReha/ProvCTDG/master/darpa_labelling/groundtruth/TC3_${labelfile}_final_aggregated.csv"
    filename="${url##*/}"

    curl -Ls -o "trace-theia/labels/$filename" "$url"
done

echo "- success."