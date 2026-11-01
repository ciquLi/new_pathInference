"""Use the frozen PATS numerical generator for target-only traceroute queries."""
from pathlib import Path
import os

from pathveil.data.prepare import resolve_read_path
from pathveil.experiment.contracts import require, verify_files
from pathveil.utils import read_json, sha256, write_json
from . import public_tree_observed_pathgen as generator
from . import public_tree_pathgen as base
from .registered_topology import RegisteredPublicPrefixTopology


def install_generator(directory, data_manifest, contract, output):
    """Bind the identical source tree to training and prediction workers."""
    import HighFrequencyVertexBasedPathGen as hf

    directory, output = Path(directory).resolve(), Path(output).resolve()
    data = read_json(data_manifest)
    source = contract['source']
    spec_file, method_file = directory / (source + '.json'), directory / 'METHOD_FROZEN.json'
    spec = read_json(spec_file)
    require(spec['source'] == source and spec['date'] == contract['date'], 'PATS source/date differs')
    require(spec['admitted_path_set_sha256'] == contract['path_set_sha256'], 'PATS admitted library differs')
    require(spec['construction_method_sha256'] == sha256(method_file), 'PATS tree method changed')
    verify_files(spec['data_code_input_sha256'])
    nominal = (Path(data['source_data_root']) / 'data/aspath/path_by_vantage' /
               data['date'] / 'path_without_prefix/train/pathWithKey.pkl')
    index = resolve_read_path(data, nominal)
    bindings = {str(p): sha256(p) for p in (spec_file, method_file, index)}
    helpers, public, selectors = base.siblings()
    weights = {tuple(r['prefix']): float(r.get('structural_weight', 1.0))
               for r in spec['prefix_classification']}
    identity = dict(date=data['date'], spec_sha256=sha256(spec_file), method_sha256=sha256(method_file))
    selector = selectors.PublicHeadSelector.from_records(source, spec['prefix_classification'], identity, weights)
    topology, calls = None, 0

    def generate(env, k, requested_source, destinations):
        nonlocal topology, calls
        require(str(requested_source) == source, 'PATS worker requested another source')
        verify_files(bindings)
        if topology is None:
            topology = RegisteredPublicPrefixTopology.from_manifest(data_manifest)
        require(topology.graph_identity_sha256 == spec['graph_identity_sha256'], 'PATS topology changed')
        library = hf.load_local_knowledge_paths(env, source)
        from pathveil.traceroute.common import path_signature
        require(len(library) == contract['rows'] and path_signature(library) == contract['path_set_sha256'],
                'PATS local knowledge differs from the admitted training labels')
        env.augment_graph_for_external_nodes(destinations)
        calls += 1
        trace = output / 'pats' / (str(os.getpid()) + '_' + str(calls) + '.jsonl')
        trace.parent.mkdir(parents=True, exist_ok=True)
        with base.Heartbeat(source, 'traceroute_pats') as progress, trace.open('w') as stream:
            result, audit = generator.generate_public_candidates(
                env, k, source, destinations, spec, selector, topology, helpers,
                public.commercial_phases, hf.networkit_sp, library,
                legacy_generator=None, progress=progress, query_trace=stream)
        audit.update(source=source, date=data['date'], bindings_sha256=bindings,
            query_trace_sha256=sha256(trace), admitted_path_set_sha256=contract['path_set_sha256'],
            generator_code_sha256=sha256(generator.__file__))
        write_json(trace.with_suffix('.json'), audit)
        return result

    hf.generate_kpath_nosave = generate
    return bindings
