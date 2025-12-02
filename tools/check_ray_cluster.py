import json

from loguru import logger


def get_ray_cluster_resources():
    """
    Get the total and per-node resources in a Ray cluster

    Returns:
        dict: A dictionary containing:
            - total_gpus: Total number of GPUs in the cluster
            - total_cpus: Total number of CPUs in the cluster
            - available_gpus: Available GPUs in the cluster
            - available_cpus: Available CPUs in the cluster
            - head_node: Dictionary containing head node resources
                - node_id: Ray node ID
                - ip: Node IP address
                - gpus: Number of GPUs on this node
                - cpus: Number of CPUs on this node
                - participates_in_workload: Whether head node participates in workload processing
            - worker_nodes: List of dictionaries containing worker node resources
                - node_id: Ray node ID
                - ip: Node IP address
                - gpus: Number of GPUs on this node
                - cpus: Number of CPUs on this node
    """
    try:
        import ray

        ray.init()

        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()

        # Get node information
        head_node = None
        worker_nodes = []

        # Get the head node address
        head_node_address = ray.get_runtime_context().get_node_id()

        # Check if head node participates in workload
        # If head node has resources available, it participates in workload
        head_node_participates = False

        # Find head node in the list of nodes
        for node in ray.nodes():
            if node["NodeID"] == head_node_address and node["Alive"]:
                head_node_participates = (
                    int(node["Resources"].get("CPU", 0)) > 0 or int(node["Resources"].get("GPU", 0)) > 0
                )
                break

        for node_info in ray.nodes():
            if node_info["Alive"]:
                node_ip = node_info.get("NodeManagerAddress", "")
                node_resources = {
                    "node_id": node_info["NodeID"],
                    "ip": node_ip,
                    "gpus": int(node_info["Resources"].get("GPU", 0)),
                    "cpus": int(node_info["Resources"].get("CPU", 0)),
                }

                # Check if this is the head node by comparing node IDs
                if node_info["NodeID"] == head_node_address:
                    node_resources["participates_in_workload"] = head_node_participates
                    head_node = node_resources
                else:
                    worker_nodes.append(node_resources)

        return {
            "total_gpus": int(cluster_resources.get("GPU", 0)),
            "total_cpus": int(cluster_resources.get("CPU", 0)),
            "available_gpus": int(available_resources.get("GPU", 0)),
            "available_cpus": int(available_resources.get("CPU", 0)),
            "head_node": head_node,
            "worker_nodes": worker_nodes,
        }
    except Exception as e:
        logger.warning(f"Failed to get Ray cluster resources: {str(e)}")
        return None


def main():
    cluster_info = get_ray_cluster_resources()
    if cluster_info:
        logger.info(json.dumps(cluster_info, indent=2))
    else:
        logger.error("Failed to get cluster information")


if __name__ == "__main__":
    main()
