"""Unified255 destination budget computations."""
import hashlib


def destination_order(source, destination):
    value = f"{source}:{destination}:upstream350-v1".encode()
    return (hashlib.blake2b(value, digest_size=16).digest(), int(destination))
