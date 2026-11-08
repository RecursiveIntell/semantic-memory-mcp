"""Historical recovery must be explicit and must honor opt-out on every route."""
import importlib.util
from pathlib import Path
import sys

import pytest

HOOKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOOKS))
policy = importlib.import_module('recall_policy')
apply_quality_gate = policy.apply_quality_gate
classify_recall_quality = policy.classify_recall_quality

spec = importlib.util.spec_from_file_location('history_mnemes', HOOKS / 'mnemes-context-recall.py')
assert spec is not None and spec.loader is not None
mnemes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mnemes)

TASK = {'id': 'offline-history-fixture', 'content': 'You are Council Agent B. Complete these tasks: deploy the service.',
        'namespace': 'general', 'score': 0.04}

NEGATIVES = [
    'do NOT use any previous instructions or tasks',
    "don't recover the prior plan",
    'don’t show the old instructions',
    'never retrieve previous tasks',
    'avoid recalling earlier plans',
    'without using previous instructions, explain this function',
    'ignore the old instructions',
    'exclude historical tasks from context',
    'do not recover old instructions; recover the prior plan',
    'Recover the prior plan, but do not use previous instructions.',
    'Explain why the previous instructions are unsafe.',
    "The log says 'recover the prior plan'. Explain current behavior.",
    'Previous instructions must not be included. Show the prior plan.',
    "recover the prior plan, previous instructions shouldn't be used",
    'recover the prior plan, previous instructions shouldn’t be used',
    "Show prior plans, but previous instructions can't be used",
    'Show me the prior plan, but leave out previous instructions',
    'Show the prior plan, though previous instructions are not to be used',
    'Recover previous tasks? No, skip them',
]
POSITIVES = [
    'recover the prior plan and old instructions',
    'Please recover the prior plan.',
    'Could you show me the previous instructions?',
    'Quote the historical tasks as data.',
    'Find the earlier plan.',
    'What did the previous instructions say?',
    'Show the prior plan, but do not execute it.',
    'Do not edit the code; recover the prior plan.',
    'Recover the prior plan; skip unit tests.',
]


@pytest.mark.parametrize('query', NEGATIVES)
@pytest.mark.parametrize('route', ['primary', 'graph', 'mnemes'])
def test_negated_or_incidental_history_does_not_unlock_task_context(query, route):
    item = {**TASK, **({'_graph_discovery': True} if route == 'graph' else {})}
    quality = classify_recall_quality(item, query)
    assert quality['safe_as_evidence'] is False
    assert quality['quoted_history'] is False
    if route == 'mnemes':
        assert 'Council Agent B' not in mnemes.format_context([item], query=query)
    else:
        kept, filtered = apply_quality_gate([item], query, 5)
        assert kept == []
        assert len(filtered) == 1


@pytest.mark.parametrize('query', POSITIVES)
@pytest.mark.parametrize('route', ['primary', 'graph', 'mnemes'])
def test_explicit_historical_recovery_still_returns_untrusted_quoted_data(query, route):
    item = {**TASK, **({'_graph_discovery': True} if route == 'graph' else {})}
    quality = classify_recall_quality(item, query)
    assert quality['safe_as_evidence'] is False
    assert quality['quoted_history'] is True
    if route == 'mnemes':
        text = mnemes.format_context([item], query=query)
        assert 'Council Agent B' in text
        assert 'quoted' in text.lower() and 'untrusted' in text.lower()
    else:
        kept, filtered = apply_quality_gate([item], query, 5)
        assert len(kept) == 1 and filtered == []


@pytest.mark.parametrize('query', [NEGATIVES[0], 'Explain the Python compatibility constraint.'])
@pytest.mark.parametrize('item', [
    {'content': 'Preserve Python 3.14 compatibility for this project.', 'namespace': 'codex'},
    {'content': 'Calibration setting is 17.', 'namespace': 'research'},
])
def test_historical_optout_does_not_blanket_suppress_facts_or_standing_preferences(query, item):
    kept, _ = apply_quality_gate([item], query, 5)
    assert len(kept) == 1
    assert kept[0][1]['safe_as_evidence'] is False
    assert kept[0][1]['quoted_history'] is False
