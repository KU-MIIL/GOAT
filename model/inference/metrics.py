"""SA / IA evaluation for API-agent predictions.

Metrics (GOAT):
- SA (API Selection Accuracy): Jaccard similarity between the predicted and
  ground-truth *sets of API functions* (function identity only).
- IA (API Invocation Accuracy): same as SA, but an element matches only when the
  function AND all of its arguments match.

Each instance is scored with Jaccard similarity, then averaged over the dataset.
Predictions and ground truth are aligned by index (the prediction file is the
inference output over the same evaluation set).
"""

import argparse
import json


def _load(path):
    with open(path, 'r') as f:
        return json.load(f)


def _api_selection_set(api_path):
    """Set of API functions (identified by docid)."""
    return {str(step['docid']) for step in api_path}


def _api_invocation_set(api_path):
    """Set of (function, arguments) invocations; arguments canonicalized to a string."""
    return {
        (str(step['docid']), json.dumps(step.get('llm_generation', {}), sort_keys=True, ensure_ascii=False))
        for step in api_path
    }


def jaccard(pred, gt):
    union = pred | gt
    if not union:
        return 1.0  # both empty -> perfect match
    return len(pred & gt) / len(union)


def evaluate(pred_data, gt_data):
    if len(pred_data) != len(gt_data):
        raise ValueError(f"prediction/ground-truth length mismatch: {len(pred_data)} vs {len(gt_data)}")

    sa_scores, ia_scores = [], []
    for pred, gt in zip(pred_data, gt_data):
        pred_path, gt_path = pred['api_path'], gt['api_path']
        sa_scores.append(jaccard(_api_selection_set(pred_path), _api_selection_set(gt_path)))
        ia_scores.append(jaccard(_api_invocation_set(pred_path), _api_invocation_set(gt_path)))

    n = len(sa_scores)
    return {
        'SA': sum(sa_scores) / n if n else 0.0,
        'IA': sum(ia_scores) / n if n else 0.0,
        'num_instances': n,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute SA (API Selection Accuracy) and IA (API Invocation Accuracy)")
    parser.add_argument('--pred_path', type=str, required=True, help='inference predictions json')
    parser.add_argument('--gt_path', type=str, required=True, help='ground-truth dataset json (e.g. seen_intertool.json)')
    args = parser.parse_args()

    result = evaluate(_load(args.pred_path), _load(args.gt_path))
    print(f"SA (API Selection Accuracy):  {result['SA'] * 100:.2f}")
    print(f"IA (API Invocation Accuracy): {result['IA'] * 100:.2f}")
    print(f"# instances: {result['num_instances']}")
