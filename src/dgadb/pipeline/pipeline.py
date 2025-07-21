from .load_graph import load_graph
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

graph = load_graph("yelp-zip")
print(graph)
