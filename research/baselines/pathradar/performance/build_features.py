"""Build optional native feature tables independently of live verified binaries."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

HERE=Path(__file__).resolve().parent
OUTPUT=HERE.parents[3]/'runs/baselines/pathradar/performance/features_build'


def build():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source=HERE/'features_native.cpp'
    binary=OUTPUT/('_pathradar_features'+sysconfig.get_config_var('EXT_SUFFIX'))
    command=['g++','-O3','-std=c++17','-shared','-fPIC','-Wall','-Wextra',
             '-Wno-missing-field-initializers','-I'+sysconfig.get_paths()['include'],str(source),'-o',str(binary)]
    subprocess.run(command,check=True)
    record=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                binary=str(binary),python=sys.version,command=command)
    (OUTPUT/'BUILD.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    build()
