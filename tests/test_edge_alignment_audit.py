"""Reject misleading architecture/initialization declarations independently."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / 'tools/validation'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('edge_alignment_audit', DIRECTORY / 'audit_edge_alignment.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


@pytest.mark.parametrize('change', ['mode', 'warm_start', 'base_initial', 'full_initial', 'head', 'active_base'])
def test_audit_rejects_hidden_architecture_or_initialization_changes(change):
    scope = dict(use_base=False, input_semantics_version=2, final_correction_head_zero=True,
        pretrained_checkpoint_loaded=False, full_initial_sha256='new', base_initial_sha256='base',
        dormant_modules=['pr', 'pm'], trainable_parameter_names=['asEmbedding.weight', 'alignment.output.3.weight'])
    model = dict(initial_sha256='new'); initial = dict(state_dict_sha256='new'); base = dict(state_dict_sha256='base')
    audit.check_architecture(scope, model, initial, base, 'edge_only')
    if change == 'mode': scope['use_base'] = True
    elif change == 'warm_start': scope['pretrained_checkpoint_loaded'] = True
    elif change == 'base_initial': scope['base_initial_sha256'] = 'trained'
    elif change == 'full_initial': model['initial_sha256'] = 'different'
    elif change == 'head': scope['final_correction_head_zero'] = False
    else: scope['trainable_parameter_names'].append('pr.ffn1.weight')
    with pytest.raises(ValueError):
        audit.check_architecture(scope, model, initial, base, 'edge_only')
