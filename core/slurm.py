from dataclasses import dataclass
import os


@dataclass
class SlurmEnv:
    rank: int
    local_rank: int
    world_rank: int
    global_rank: int
    world_size: int
    node_rank: int
    num_nodes: int
    ntasks_per_node: int
    node_list: list[str]
    job_id: str
    is_slurm: bool


def get_slurm_env() -> SlurmEnv:
    """Get SLURM environment variables for distributed processing.

    Returns:
        Dictionary containing SLURM environment variables:
        - rank: Local rank within the node (SLURM_LOCALID or SLURM_PROCID % SLURM_NTASKS_PER_NODE)
        - local_rank: Local rank within the node (SLURM_LOCALID)
        - world_rank: Global rank across all nodes (SLURM_PROCID)
        - world_size: Total number of tasks (SLURM_NTASKS)
        - node_rank: Rank of the node (SLURM_NODEID)
        - num_nodes: Number of nodes (SLURM_NNODES)
        - ntasks_per_node: Number of tasks per node (SLURM_NTASKS_PER_NODE)
        - node_list: List of node names (SLURM_NODELIST)
        - job_id: Job ID (SLURM_JOB_ID)
        - is_slurm: Whether we're running under SLURM
    """
    # Initialize with default values
    env = SlurmEnv(
        rank=0,
        local_rank=0,
        world_rank=0,
        global_rank=0,
        world_size=1,
        node_rank=0,
        num_nodes=1,
        ntasks_per_node=1,
        node_list=[],
        job_id="",
        is_slurm=False,
    )

    # Check if we're in a SLURM environment
    env.is_slurm = "SLURM_JOB_ID" in os.environ

    if not env.is_slurm:
        # Not in SLURM, return defaults
        return env

    # Extract SLURM environment variables
    env.world_rank = int(os.environ.get("SLURM_PROCID", 0))
    env.world_size = int(os.environ.get("SLURM_NTASKS", 1))
    env.node_rank = int(os.environ.get("SLURM_NODEID", 0))
    env.num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    env.ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
    env.local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    env.node_list = os.environ.get("SLURM_NODELIST", "").split(",")
    env.job_id = os.environ.get("SLURM_JOB_ID", "")  # Provide default empty string

    # For rank within node, prefer SLURM_LOCALID, fallback to calculating from PROCID
    if "SLURM_LOCALID" in os.environ:
        env.rank = env.local_rank
    else:
        # Calculate local rank from global rank and node information
        env.rank = env.world_rank % env.ntasks_per_node

    return env
