"""An independent audit must reject changed checkpoints and out-of-set selections."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

_FILE = Path(__file__).resolve().parents[1] / 'tools/validation/audit_gmpi_confidence.py'
_SPEC = importlib.util.spec_from_file_location('audit_gmpi_confidence', _FILE)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def test_checkpoint_audit_rejects_test_selected_or_nonbest_epoch():
    history = [dict(epoch=i, training_jaccard=.8, validation_jaccard=score,
                    training_loss=.2, validation_original_smooth_l1_loss=.1,
                    wall_seconds=2., cpu_seconds=1., checkpoint_improved=i < 2)
               for i, score in enumerate([.8, .9, .85])]
    model = dict(epochs=3, selected_epoch=1, validation_jaccard=.9)
    audit.check_history(history, model, 2)
    with pytest.raises(ValueError, match='Selected checkpoint'):
        audit.check_history(history, dict(model, selected_epoch=2), 2)
    with pytest.raises(ValueError, match='patience'):
        audit.check_history(history + [dict(history[-1], epoch=3)], dict(model, epochs=4), 1)


def test_membership_audit_checks_slot_cap_and_direct_queries(tmp_path):
    # Slot 0 has wrong endpoints and must not consume the usable-candidate cap.
    vocabulary = ['<pad>', '1', '2', '3', '4']
    candidate = tmp_path / 'candidates.csv'
    with candidate.open('w', newline='') as stream:
        csv.writer(stream).writerow(['2->1|1->3->2|1->3->2|1->4->2', '', '', '0.0|0.0|0.0|0.0', '1->2'])
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps(dict(pairs=['1->2', '1->3', '1->4'])))
    dispatch = tmp_path / 'dispatch.jsonl'
    states = ['model_candidates', 'direct_graph_edge_candidate', 'no_usable_candidate']
    dispatch.write_text(''.join(json.dumps(dict(pair=p, status=s)) + '\n'
                                for p, s in zip(['1->2', '1->3', '1->4'], states)))
    binding = dict(candidate=str(candidate), inventory=str(inventory), dispatch=str(dispatch))
    prediction = tmp_path / 'predictions.csv'

    def write(selected):
        with prediction.open('w', newline='') as stream:
            csv.writer(stream).writerows([
                ['pair', 'predicted_path', 'status'], ['1->2', selected, 'predicted'],
                ['1->3', '1->3', 'direct_graph_edge'], ['1->4', '', 'no_usable_candidate']])

    write('1->3->2')
    checked = audit.candidate_membership(binding, vocabulary, {'loss': prediction}, 1)
    assert checked['queries'] == 3
    assert checked['maximum_usable_candidates'] == 2
    assert checked['slot_capped_queries'] == 1
    write('1->4->2')
    with pytest.raises(ValueError, match='outside frozen candidates'):
        audit.candidate_membership(binding, vocabulary, {'loss': prediction}, 1)
    checked = audit.candidate_membership(binding, vocabulary, {'loss': prediction}, 2)
    assert checked['status'] == 'PASS'
    assert checked['slot_capped_queries'] == 0
    checked = audit.candidate_membership(binding, vocabulary, {'loss': prediction}, None)
    assert checked['status'] == 'PASS'
    assert checked['slot_capped_queries'] == 0
    assert checked['candidate_slots'] is None


def test_membership_audit_rejects_nonzero_final_labels():
    with pytest.raises(ValueError, match='contains labels'):
        audit.usable_candidates(['1->2', '', '', '1.0', '1->2'], ['<pad>', '1', '2'], 200)
