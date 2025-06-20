#!/usr/bin/bash

set -e

echo "DGraph"

echo "- Formatting data..."

python scripts/dgraph.py
rm data/dgraph/dgraphfin.npz data/dgraph/dgraphfinv2_edge_timestamp.npy data/dgraph/dgraphfinv2_node_timestamp.npy

echo "- success."