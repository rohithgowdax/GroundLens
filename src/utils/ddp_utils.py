import os
import torch
import torch.distributed as dist


def setup_ddp():
    """
    Initialize distributed training process group.

    Returns:
        tuple: (rank, local_rank, world_size)
            - rank: Global rank of this process
            - local_rank: Local rank on this node
            - world_size: Total number of processes
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # Set device before initializing process group
        torch.cuda.set_device(local_rank)

        # Initialize process group with NCCL backend for GPU training
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{local_rank}")
        )

    return rank, local_rank, world_size


def cleanup_ddp():
    """Clean up distributed training process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is the main process (rank 0)."""
    return rank == 0


def reduce_metric(value, world_size, device):
    """
    Reduce a metric across all processes by averaging.

    Args:
        value: Metric value to reduce (float or torch.Tensor)
        world_size: Total number of processes
        device: Device to perform reduction on

    Returns:
        float: Averaged metric value across all processes
    """
    if world_size <= 1:
        return value

    # Convert to tensor if needed
    if not torch.is_tensor(value):
        value = torch.tensor(value, device=device)
    else:
        value = value.to(device)

    # All-reduce sum across all processes
    dist.all_reduce(value, op=dist.ReduceOp.SUM)

    # Average by dividing by world_size
    value = value.item() / world_size

    return value