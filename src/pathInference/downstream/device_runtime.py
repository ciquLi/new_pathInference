# Extracted from the frozen 2026-10-24 unified-pool run.
# Numerical training/evaluation logic preserved; date/run/core imports adapted.
"""Isolated CPU routing for unchanged legacy GMPI ``.cuda()`` calls.

The original trainer is imported unchanged. On CPU only, its device-transfer
calls are routed to CPU; no model, loss, batch, or optimizer code is replaced.
"""
from __future__ import annotations


def configure_device(torch, requested="auto", threads=4):
    if requested not in {"auto", "cpu", "cuda"} or not 1 <= threads <= 16:
        raise ValueError("Invalid device/thread configuration")
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("Explicit CUDA device requested but unavailable")
    kind = "cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu"
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A test may have initialized the inter-op pool before this function.
        if torch.get_num_interop_threads() != 1:
            raise
    if kind == "cpu":

        def tensor_cpu(self, device=None, non_blocking=False, memory_format=None):
            options = {"device": "cpu", "non_blocking": non_blocking}
            if memory_format is not None:
                options["memory_format"] = memory_format
            return self.to(**options)

        def module_cpu(self, device=None):
            return self.to("cpu")

        torch.Tensor.cuda = tensor_cpu
        torch.nn.Module.cuda = module_cpu
    return torch.device(kind), dict(
        requested=requested,
        effective=kind,
        cuda_available=available,
        intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        cpu_routing="legacy Tensor.cuda/Module.cuda -> to(cpu)" if kind == "cpu" else None,
        original_core_source_files_modified=False,
    )
