"""Integer-graph shortest traversal; original XGBoost prediction is unchanged."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import MethodType

import native_engine as bridge
from build_shortest import OUTPUT

record=json.loads((OUTPUT/'BUILD.json').read_text())
for path,key in [(Path(__file__).with_name('shortest_native.cpp'),'source_sha256'),(Path(record['binary']),'binary_sha256')]:
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record[key]:
        raise ValueError('Shortest-path build identity changed: '+str(path))
sys.path.insert(0,str(OUTPUT))
import _pathradar_shortest as native


def shortest(sample,asn):
    return native.shortest_stream(sample._integer_graph,sample,asn)


class IntegerSample(bridge.engine.original.Model_Sample):
    def simulate(self,asn):
        # Only shortest lengths enter candidate generation and features. Materialize
        # auxiliary diagnostic paths on demand, with the exact original ordering.
        self._shortest_source=asn
        for name in ('AS_path','worst','path_flag'):
            self.__dict__.pop(name,None)
        self.shortest=native.shortest_stream(self._integer_graph,self,asn,True)

    def __getattr__(self,name):
        if name not in ('AS_path','worst','path_flag') or '_shortest_source' not in self.__dict__:
            raise AttributeError(name)
        asn=self._shortest_source
        self.AS_path=defaultdict(list,{asn:[asn]})
        self.worst=defaultdict(list,{asn:[asn]})
        self.path_flag={(asn,):0}
        native.shortest_stream(self._integer_graph,self,asn)
        return self.__dict__[name]


def enable_sample(sample):
    sample._integer_graph=native.pack(sample)
    sample.__class__=IntegerSample
    sample.stream=MethodType(shortest,sample)
    bridge.engine.index_link_order(sample)


class IntegerTree(bridge.NativeTree):
    pass


def main():
    bridge.engine.PersistentTree=IntegerTree
    bridge.engine.accelerate_sample=enable_sample
    bridge.engine.main()


if __name__=='__main__':
    main()
