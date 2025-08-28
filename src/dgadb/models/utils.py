from time import perf_counter
from functools import wraps
import logging
logger = logging.getLogger(__name__)

def time_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' took {total_time:.4f} seconds")
        return result
    return wrapper