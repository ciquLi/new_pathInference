"""C++ engine wrapper for knownpath BFS computation."""
import ctypes
import json
import tempfile
import os
from pathlib import Path

_LIB = ctypes.CDLL(str(Path(__file__).resolve().parent / 'libknownpath.so'))

_LIB.engine_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_LIB.engine_create.restype = ctypes.c_void_p

_LIB.engine_destroy.argtypes = [ctypes.c_void_p]
_LIB.engine_destroy.restype = None

_LIB.engine_compute.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_LIB.engine_compute.restype = ctypes.c_void_p

_LIB.engine_free_result.argtypes = [ctypes.c_void_p]
_LIB.engine_free_result.restype = None


def _dump_to_temp(data, prefix):
    fd, path = tempfile.mkstemp(suffix='.json', prefix=prefix)
    os.write(fd, json.dumps(data).encode())
    os.close(fd)
    return path


class KnownpathEngine:
    def __init__(self, env):
        db_path = os.environ.get('KNOWNPATH_TRAIN_DB') or env.baseDir.replace('RL_Graph/', 'path_without_prefix/train/knownpath.db')
        self._temps = []

        # Dump pre-parsed dicts to clean JSON (avoid ujson double-encoding issues)
        edge_path = _dump_to_temp(env.edge_freq_raw, 'knownpath_edge_')
        rel_path = _dump_to_temp(env.rel_dict, 'knownpath_rel_')
        self._temps = [edge_path, rel_path]

        self._handle = _LIB.engine_create(
            db_path.encode(),
            edge_path.encode(),
            rel_path.encode()
        )

    def close(self):
        if self._handle:
            _LIB.engine_destroy(self._handle)
            self._handle = None
        for p in self._temps:
            try:
                os.unlink(p)
            except OSError:
                pass
        self._temps.clear()

    def compute(self, sure_paths, base_as):
        import ujson
        payload = ujson.dumps({'sure_paths': sure_paths, 'base_as': base_as})
        c_result = _LIB.engine_compute(self._handle, payload.encode())
        raw = ctypes.cast(c_result, ctypes.c_char_p).value.decode()
        _LIB.engine_free_result(c_result)
        return ujson.loads(raw)
