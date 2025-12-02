# The Star-Graph-Connected-Components (SGCC) algorithm here referenced from:
# https://github.com/bigcode-project/bigcode-dataset/blob/main/near_deduplication/minhash_deduplication_spark.py
# --------------------------------------------------------

from typing import List, Optional, Tuple

from loguru import logger
from pyspark import SparkConf
from pyspark.sql import SparkSession


def init_spark(
    master_url: Optional[str] = None,
    spark_executor_memory=None,
    spark_driver_memory=None,
    spark_executor_memoryOverhead=None,
):
    if not spark_executor_memory:
        spark_executor_memory = "64g"
    if not spark_driver_memory:
        spark_driver_memory = "64g"
    if not spark_executor_memoryOverhead:
        spark_executor_memoryOverhead = "20000"
    if not master_url:
        master_url = "local[*]"
    conf = SparkConf()
    conf.set("spark.app.name", "MinHashLSH")
    conf.set("spark.debug.maxToStringFields", "100")
    conf.set("spark.master", master_url)
    conf.set("spark.executor.memory", spark_executor_memory)
    conf.set("spark.driver.memory", spark_driver_memory)
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    conf.set("spark.executor.memoryOverhead", spark_executor_memoryOverhead)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    logger.info("Spark initialization done.")
    return spark


def generate_edges(nodes: List[int]) -> List[Tuple[int, int]]:
    """
    Generate edges from a cluster. Instead of generating N^2 edges,
    we only need all nodes align to a single node,
    since we will be running connected components on the edges later.

    Parameters
    ----------
    nodes : List[int]
        The list of nodes in the cluster.

    Returns
    -------
    List[Tuple[int, int]]
        The list of edges.
    """
    if len(nodes) <= 1:
        return []

    min_node = min(nodes)
    return [(n, min_node) for n in nodes if n != min_node]


# Connected Components in MapReduce and Beyond
def large_star_map(edge):
    return [(edge[0], edge[1]), (edge[1], edge[0])]


def large_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n > x]


def small_star_map(edge):
    x, y = edge
    if y <= x:
        return (x, y)
    else:
        return (y, x)


def small_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n != minimum]


def find_components(edges):
    """
    Star-Graph-Connected-Components (SGCC) algorithm
    """

    a = edges
    while True:
        b = a.flatMap(large_star_map).groupByKey().flatMap(large_star_reduce).distinct().cache()
        a = b.map(small_star_map).groupByKey().flatMap(small_star_reduce).distinct().cache()
        changes = a.subtract(b).union(b.subtract(a)).collect()
        if len(changes) == 0:
            break

    results = a.collect()
    return results
