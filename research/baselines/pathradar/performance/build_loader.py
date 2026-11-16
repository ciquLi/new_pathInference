"""Build the streaming evidence loader independently of verified live binaries."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[3]/'runs/baselines/pathradar/performance/loader_build'


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = HERE/'loader_native.cpp'
    binary = OUTPUT/('_pathradar_loader'+sysconfig.get_config_var('EXT_SUFFIX'))
    command = ['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra',
               '-Wno-missing-field-initializers', '-I'+sysconfig.get_paths()['include'],
               str(source), '-o', str(binary)]
    subprocess.run(command, check=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    record = dict(source_sha256=digest(source), binary_sha256=digest(binary), binary=str(binary),
                  dependencies_sha256={str(HERE/name): digest(HERE/name)
                      for name in ('pipeline_native.cpp', 'features_native.cpp')},
                  python=sys.version, command=command)
    (OUTPUT/'BUILD.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    build()
