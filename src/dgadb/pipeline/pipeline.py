import logging
from src.dgadb.preprocessing.pipeline import Pipeline
from src.dgadb.preprocessing.anomaly_injector import AnomalyInjector


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

if __name__ == "__main__":
    config_name = "yelp-zip-example"
    # config_name = "bitcoin-alpha-example"

    pipeline = Pipeline.from_config(config_name, force_rerun=False)

    g = pipeline.run()
    g.describe()

    temporal_graph = g.to_temporal_graph()
    temporal_graph.describe()

    # temporal_graph.msg = temporal_graph.msg.to("mps")
    # print(f'{temporal_graph.msg.device}')

    # TODO @Dastamn: Test on GPU
    anom_injector = AnomalyInjector(temporal_graph)

    anom_injector.generate_anomalous_edges("temporal", anom_test_ratio=0.15)
