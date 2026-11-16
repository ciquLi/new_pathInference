"""Native hot loops with original PathRadar models, thresholds and containers."""
import hashlib
import json
from pathlib import Path
import sys
from types import MethodType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine
from build_native import OUTPUT


def load_extension():
    record = json.loads((OUTPUT/'BUILD.json').read_text())
    for path,key in [(HERE/'native.cpp','source_sha256'),(Path(record['binary']),'binary_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record[key]:
            raise ValueError('Native source/binary identity changed: '+str(path))
    sys.path.insert(0,str(OUTPUT))
    import _pathradar_native
    return _pathradar_native


native = load_extension()


def native_shortest(self,asn):
    return native.shortest_stream(self,asn)


def enable(tree):
    engine.index_link_order(tree.ms)
    tree.ms.stream = MethodType(native_shortest,tree.ms)
    return tree


class NativeTree(engine.PersistentTree):
    def get_model(self,paths,model_name):
        atts=engine.original.np.array(native.features(self.ms,paths))
        predict=self.model[model_name].predict_proba(atts)
        return [value[1] for value in predict]

    def stream(self,asn):
        active=[(asn,)]
        candidate=set()
        for _ in range(9):
            paths=native.expand(self,active)
            candidate=native.add_paths(self,paths,candidate)
            active=list(candidate)
            active.sort(key=lambda path:int(path[-1]))
            candidate=set()


def main():
    # Keep the previously verified execution adapter unmodified on disk.
    engine.PersistentTree=NativeTree
    engine.accelerate_sample=lambda sample: setattr(sample,'stream',MethodType(native_shortest,sample)) or engine.index_link_order(sample)
    engine.main()


if __name__=='__main__':
    main()
