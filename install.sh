#!/bin/bash

export BASE_PATH=$(pwd)

echo "Creating venv..."
virtualenv venv
source venv/bin/activate

echo "Installing pip dependencies..."
pip install -r requirements.txt

echo "Installing StrGNN dependencies..."
cd src/dgadb/models/StrGNN/pytorch_DGCNN
cd lib
make clean
make -j4

echo "...done!"
