import logging

from src.dgadb.preprocessing.pipeline import Pipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


if __name__ == "__main__":
    config_name = "yelp-zip-example"

    pipeline = Pipeline.from_config(config_name, force_rerun=False)

    g = pipeline.run()
    g.describe()
