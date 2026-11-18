"""What machine a measurement was taken on.

Module 4 turns on numbers that are meaningless without their hardware: tokens
per second, VRAM headroom and the latency of a 4-bit kernel are properties of a
(model, runtime, device) triple, not of a model. A report that states "81 ms
p95" without saying on what has recorded an anecdote.

So every benchmark in this project writes this block alongside its results, and
it is captured programmatically rather than typed into the report, because a
hand-written hardware note is exactly the kind of thing that stays unchanged
when the machine does.

``torch`` is imported lazily. This module is also read by the quantization
scripts, which run in a *separate* interpreter with a different torch version
(see reports/module-4.md), and a hard import would make the hardware block
unobtainable in the one place the GPU question matters most.
"""

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

# nvidia-smi rather than torch.cuda: the point is to record the GPU that is
# physically present even when the installed torch cannot use it, which is
# precisely this machine's situation and the central constraint of Module 4.
_SMI_QUERY = "name,memory.total,driver_version,compute_cap"


@dataclass(frozen=True)
class GpuInfo:
    """One CUDA device as the driver reports it, not as torch sees it."""

    name: str
    memory_total_mib: int
    driver_version: str
    compute_capability: str


@dataclass(frozen=True)
class Hardware:
    """The machine, the runtime and whether the GPU is actually reachable.

    ``torch_cuda_available`` is separate from ``gpus`` on purpose. A box can
    have a CUDA device and a CPU-only torch build, in which case every 4-bit
    kernel in the ecosystem is unavailable while ``nvidia-smi`` still prints a
    healthy card. Collapsing the two into one "has GPU" boolean is how a
    report ends up claiming a measurement the interpreter could not have made.
    """

    platform: str
    processor: str
    python_version: str
    logical_cores: int
    ram_total_gib: float
    torch_version: str = "not installed"
    torch_cuda_available: bool = False
    torch_cuda_build: str | None = None
    torch_num_threads: int | None = None
    gpus: list[GpuInfo] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """One line for a report table or a log."""
        gpu = self.gpus[0].name if self.gpus else "no CUDA device"
        vram = f"{self.gpus[0].memory_total_mib} MiB" if self.gpus else "n/a"
        reachable = "reachable" if self.torch_cuda_available else "NOT reachable"
        return (
            f"{self.processor}, {self.logical_cores} logical cores, "
            f"{self.ram_total_gib:.1f} GiB RAM; GPU {gpu} ({vram}, {reachable} "
            f"from torch {self.torch_version})"
        )


def _ram_total_gib() -> float:
    try:
        import psutil
    except ImportError:
        return 0.0
    return psutil.virtual_memory().total / 1024**3


def _query_gpus() -> list[GpuInfo]:
    """Ask the driver what cards exist. An absent nvidia-smi means none."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []

    gpus: list[GpuInfo] = []
    for line in completed.stdout.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 4:
            continue
        name, memory, driver, capability = fields[:4]
        gpus.append(
            GpuInfo(
                name=name,
                # "4096 MiB" -> 4096. Keep the unit out of the number so the
                # VRAM arithmetic in the report can be done on it.
                memory_total_mib=int(memory.split()[0]) if memory.split() else 0,
                driver_version=driver,
                compute_capability=capability,
            )
        )
    return gpus


def describe_hardware() -> Hardware:
    """Capture the machine a benchmark is about to run on."""
    torch_version = "not installed"
    cuda_available = False
    cuda_build: str | None = None
    num_threads: int | None = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        cuda_build = torch.version.cuda
        num_threads = torch.get_num_threads()
    except ImportError:
        pass

    return Hardware(
        platform=platform.platform(),
        processor=platform.processor() or platform.machine(),
        python_version=platform.python_version(),
        logical_cores=os.cpu_count() or 0,
        ram_total_gib=_ram_total_gib(),
        torch_version=torch_version,
        torch_cuda_available=cuda_available,
        torch_cuda_build=cuda_build,
        torch_num_threads=num_threads,
        gpus=_query_gpus(),
    )


def hardware_json(indent: int = 2) -> str:
    return json.dumps(describe_hardware().as_dict(), indent=indent)
