from __future__ import annotations

import datetime
import ctypes
import json
import os
import socket
import time

import torch
import torch.distributed as dist


def required_int(name: str) -> int:
    try:
        return int(os.environ[name])
    except KeyError as error:
        raise RuntimeError(f"{name} is required") from error


def runtime_nccl_version() -> int:
    library = ctypes.CDLL("libnccl.so.2")
    version = ctypes.c_int()
    if library.ncclGetVersion(ctypes.byref(version)) != 0:
        raise RuntimeError("ncclGetVersion failed")
    return int(version.value)


def main() -> None:
    rank = required_int("RANK")
    world_size = required_int("WORLD_SIZE")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    timeout_seconds = int(os.environ.get("IAPL_DISTRIBUTED_TIMEOUT_SECONDS", "7200"))
    delay_rank = int(os.environ.get("IAPL_PROBE_DELAY_RANK", "-1"))
    delay_seconds = float(os.environ.get("IAPL_PROBE_DELAY_SECONDS", "0"))
    if timeout_seconds <= 0 or delay_seconds < 0:
        raise ValueError("Probe timeout must be positive and delay must be non-negative")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )
    if rank == delay_rank:
        time.sleep(delay_seconds)

    value = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(value)
    torch.cuda.synchronize()
    expected = world_size * (world_size + 1) / 2
    result = float(value.item())
    if result != expected:
        raise RuntimeError(f"Collective returned {result}, expected {expected}")
    print(
        json.dumps(
            {
                "host": socket.gethostname(),
                "rank": rank,
                "world_size": world_size,
                "sum": result,
                "timeout_seconds": timeout_seconds,
                "delayed": rank == delay_rank,
                "delay_seconds": delay_seconds if rank == delay_rank else 0.0,
                "torch_nccl_compile_version": torch.cuda.nccl.version(),
                "runtime_nccl_version": runtime_nccl_version(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
