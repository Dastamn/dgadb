# dgadb
Dynamic Graph Anomaly Detection Benchmark

Do the following after cloning the repo:
```
cd dgadb
export BASE_PATH=$(pwd)

virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the SLADE pipeline, do:
```
python -m src.dgadb.pipeline.pipeline_SLADE
```
