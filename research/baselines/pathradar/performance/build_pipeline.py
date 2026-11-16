"""Build candidate tables without overwriting a live verified extension."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[3]/'runs/baselines/pathradar/performance/pipeline_build'


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = HERE/'pipeline_native.cpp'
    binary = OUTPUT/('_pathradar_pipeline'+sysconfig.get_config_var('EXT_SUFFIX'))
    command = ['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra',
               '-Wno-missing-field-initializers', '-I'+sysconfig.get_paths()['include'],
               str(source), '-o', str(binary)]
    subprocess.run(command, check=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    record = dict(source_sha256=digest(source), binary_sha256=digest(binary), binary=str(binary),
                  dependencies_sha256={str(HERE/'features_native.cpp'): digest(HERE/'features_native.cpp')},
                  python=sys.version, command=command)
    (OUTPUT/'BUILD.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    build()
