"""BGP-only training, target-only prediction and independent traceroute scoring."""


def run_traceroute(run_dir, data_manifest_path, release_path, device="cuda", **kwargs):
    from .runner import run_traceroute as run

    return run(run_dir, data_manifest_path, release_path, device=device, **kwargs)


__all__ = ["run_traceroute"]
