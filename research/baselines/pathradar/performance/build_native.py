"""Build the optional C++ CPython extension without external dependencies."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = ROOT / 'runs/baselines/pathradar/performance/native_build'


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = HERE / 'native.cpp'
    extension = OUTPUT / ('_pathradar_native' + sysconfig.get_config_var('EXT_SUFFIX'))
    command = ['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra',
               '-Wno-missing-field-initializers', '-I'+sysconfig.get_paths()['include'],
               str(source), '-o', str(extension)]
    subprocess.run(command, check=True)
    record = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  binary_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                  binary=str(extension), python=sys.version, command=command)
    (OUTPUT/'BUILD.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    return extension


if __name__ == '__main__':
    build()
