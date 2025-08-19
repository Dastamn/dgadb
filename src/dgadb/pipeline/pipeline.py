import logging
from src.dgadb.preprocessing.pipeline import Pipeline
from src.dgadb.preprocessing.anomaly_injector import AnomalyInjector
from src.dgadb.storage import TemporalGraphSnapshotLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


if __name__ == "__main__":
    # config_name = "yelp-zip-example"
    config_name = "bitcoin-alpha-example"

    pipeline = Pipeline.from_config(config_name, force_rerun=False)

    g = pipeline.run()
    g.describe()

    temporal_graph = g.to_temporal_graph()
    temporal_graph.describe()

    # TODO @Dastamn: Test on GPU
    anom_injector = AnomalyInjector(temporal_graph)

    anomalous_temporal_graph = anom_injector.generate_anomalous_samples(
        "c", anom_test_ratio=0.05, anom_val_ratio=0.05)

    snapshot_loader = TemporalGraphSnapshotLoader(
        anomalous_temporal_graph, strategy="window", window_size=1000)

    print("Number of snapshots: ", len(snapshot_loader))

    for snap in snapshot_loader:
        print(snap)
