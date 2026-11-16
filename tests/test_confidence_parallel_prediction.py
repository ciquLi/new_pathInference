"""Resource handover must not create two writers for a frozen source journal."""
import importlib.util
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

FILE = Path(__file__).resolve().parents[1] / 'research/candidate_confidence_v2/parallel_predict.py'
spec = importlib.util.spec_from_file_location('confidence_parallel_prediction', FILE)
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)


def records():
    runner = Path('/repo/run.py')
    parent = dict(pid=10, state='S', parent=1, start_ticks=20, arguments=['python', str(runner)])
    child = dict(pid=11, state='R', parent=10, start_ticks=21,
                 arguments=['python', str(runner), '--stage', 'predict', '--source', '199524'])
    return runner, parent, child


@pytest.mark.parametrize('change', ['parent_stopped', 'wrong_owner', 'wrong_source', 'training'])
def test_rejects_unowned_or_different_running_work(change):
    runner, parent, child = records()
    dispatch.validate_processes(parent, child, runner, '199524')
    if change == 'parent_stopped':
        parent['state'] = 'T'
    elif change == 'wrong_owner':
        child['parent'] = 12
    elif change == 'wrong_source':
        child['arguments'][-1] = '398465'
    else:
        child['arguments'][-3] = 'train'
    with pytest.raises(ValueError):
        dispatch.validate_processes(parent, child, runner, '199524')


def test_ahead_child_is_terminal_before_original_queue_resumes():
    events = []

    class Child:
        def poll(self):
            return None

        def terminate(self):
            events.append('terminate_ahead')

        def wait(self, timeout):
            events.append('ahead_terminal')

    _, parent, _ = records()
    current = dict(parent, state='T')
    dispatch.stop_ahead_before_resume(Child(), parent, inspect=lambda _: current,
                                     send=lambda pid, sig: events.append((pid, sig)))
    assert events == ['terminate_ahead', 'ahead_terminal', (10, signal.SIGCONT)]


def test_pid_reuse_never_signals_an_unrelated_process():
    _, parent, _ = records()
    current = dict(parent, state='T', start_ticks=999)
    signals = []
    dispatch.stop_ahead_before_resume(None, parent, inspect=lambda _: current,
                                     send=lambda *args: signals.append(args))
    assert signals == []


def test_parallel_progress_retains_both_live_source_counts(tmp_path):
    from pathInference.utils import write_json, read_json
    sources = ['199524', '398465']
    for source, count in zip(sources, [10, 20]):
        write_json(tmp_path / ('as' + source) / 'STATUS.json',
                   dict(source=source, stage='predicting', committed_queries=count))
    queue = SimpleNamespace(experiment=SimpleNamespace(RUN=tmp_path, config=lambda: dict(sources=sources)),
                            receipt_path=lambda source, stage: tmp_path / (source + '_' + stage))
    _, parent, _ = records()
    dispatch.progress(queue, parent, sources[0], sources[1], SimpleNamespace(pid=12), 'parallel_predicting')
    state = read_json(tmp_path / 'PARALLEL_PREDICTION.json')
    assert [row['committed_queries'] for row in state['sources']] == [10, 20]
    assert state['cpu_affinity'] == [6, 7]
