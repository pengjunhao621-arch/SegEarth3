import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rank scale-stability rerank pairs by potential benefit.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='Pair summary CSV files from summarize_scale_stability_rerank_stats.py.')
    parser.add_argument(
        '--output',
        required=True,
        help='Output CSV path.')
    parser.add_argument(
        '--min-gate-pixels',
        type=int,
        default=1,
        help='Ignore rows with fewer gated pixels.')
    return parser.parse_args()


def as_float(value, default=0.0):
    if value is None or value == '':
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(value, default=0):
    return int(round(as_float(value, default)))


def read_rows(paths):
    for path in paths:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['_source'] = path
                yield row


def main():
    args = parse_args()
    rows = []
    for row in read_rows(args.inputs):
        gate_pixels = as_int(row.get('gate_valid_pixels'))
        if gate_pixels < args.min_gate_pixels:
            continue
        improved = as_int(row.get('improved_gate_pixels'))
        harmed = as_int(row.get('harmed_gate_pixels'))
        net = as_int(row.get('net_improved_gate_pixels'))
        cf_improved = as_int(row.get('gate_counterfactual_improved_pixels'))
        cf_harmed = as_int(row.get('gate_counterfactual_harmed_pixels'))
        cf_net = as_int(row.get('gate_counterfactual_net_pixels'))
        if cf_improved == 0 and cf_harmed == 0:
            cf_improved = as_int(row.get('gate_true_target_pixels'))
            cf_harmed = as_int(row.get('gate_true_competitor_pixels'))
            cf_net = cf_improved - cf_harmed
        out = dict(row)
        out['real_net_pixels'] = net
        out['real_precision'] = (
            '' if improved + harmed == 0 else improved / (improved + harmed))
        out['counterfactual_net_pixels'] = cf_net
        out['counterfactual_precision'] = (
            '' if cf_improved + cf_harmed == 0
            else cf_improved / (cf_improved + cf_harmed))
        out['gate_pixels'] = gate_pixels
        rows.append(out)

    rows.sort(key=lambda item: (
        as_float(item.get('counterfactual_net_pixels')),
        as_float(item.get('real_net_pixels')),
        as_float(item.get('gate_pixels')),
    ), reverse=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    preferred = [
        'dataset_name',
        'target_class_name',
        'competitor_class_name',
        'gate_pixels',
        'counterfactual_net_pixels',
        'counterfactual_precision',
        'real_net_pixels',
        'real_precision',
        'gate_true_target_ratio',
        'gate_true_competitor_ratio',
        'gate_true_other_ratio',
        'pair_error_gate_coverage',
        'requested_scale',
        'gate_name',
        'drop_threshold',
        'pair_mode',
        'query_mode',
        'min_base_margin',
        'max_base_margin',
        'min_scaled_margin',
        'max_target_rank',
        '_source',
    ]
    keys = []
    for key in preferred:
        if key not in keys:
            keys.append(key)
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
