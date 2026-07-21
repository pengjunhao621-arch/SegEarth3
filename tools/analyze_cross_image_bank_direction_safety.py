#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def load_jsonl(inputs):
    paths = []
    for pattern in inputs:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def to_float(value, default=0.0):
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    if value is None or value == '':
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def log1p(value):
    return math.log1p(max(0.0, float(value)))


def class_tokens(name):
    tokens = set()
    normalized = str(name or '').lower().replace('/', ',').replace('-', ' ')
    for item in normalized.split(','):
        item = item.strip()
        if not item:
            continue
        tokens.add(item)
        for part in item.split():
            if part:
                tokens.add(part)
    return tokens


def class_role(name):
    tokens = class_tokens(name)
    if tokens & {'background', 'other', 'clutter', 'void', 'unknown'}:
        return 'catch_all'
    if tokens & {'car', 'vehicle', 'truck', 'ship', 'airplane', 'plane'}:
        return 'vehicle'
    if tokens & {
            'building', 'roof', 'house', 'facade', 'wall',
            'construction'}:
        return 'built'
    if tokens & {'tree', 'forest', 'wood', 'canopy'}:
        return 'woody_vegetation'
    if tokens & {
            'grass', 'vegetation', 'low', 'crop', 'cropland',
            'agricultural', 'agriculture', 'farmland', 'field'}:
        return 'low_vegetation'
    if tokens & {'road', 'pavement', 'impervious', 'surface', 'sidewalk'}:
        return 'impervious_surface'
    if tokens & {'water', 'river', 'lake', 'sea', 'pond'}:
        return 'water'
    if tokens & {'bareland', 'barren', 'soil', 'sand', 'bare'}:
        return 'bareland'
    return 'other_landcover'


def pair_key(dataset, space, variant, gt_idx, pred_idx):
    return (
        str(dataset),
        str(space),
        f'leave_one_{variant}',
        int(gt_idx),
        int(pred_idx),
    )


def load_pair_summary(path):
    out = {}
    for row in read_csv(path):
        key = pair_key(
            row.get('dataset'),
            row.get('space'),
            row.get('bank_variant', '').replace('leave_one_', '', 1),
            row.get('gt_class_index'),
            row.get('pred_class_index'),
        )
        out[key] = dict(
            pixels=to_int(row.get('covered_pair_pixels')),
            margin=to_float(row.get('mean_bank_margin')),
            pos=to_float(row.get('positive_margin_pixel_ratio')),
            base_gap=to_float(row.get('mean_baseline_pred_gt_margin')),
            images=to_int(row.get('images')),
        )
    return out


def load_class_coverage(path):
    out = {}
    for row in read_csv(path):
        key = (
            str(row.get('dataset')),
            str(row.get('space')),
            int(to_int(row.get('class_index'))),
        )
        out[key] = dict(
            class_name=row.get('class_name', ''),
            images=to_int(row.get('images')),
            images_with_seed=to_int(row.get('images_with_seed')),
            image_seed_coverage=to_float(row.get('image_seed_coverage')),
            seed_pixels=to_int(row.get('seed_pixels')),
            # This is GT-auditing only. It must not be used by no-GT rules.
            audit_seed_purity=to_float(row.get('mean_seed_purity')),
        )
    return out


def aggregate_recompose(inputs):
    overall = defaultdict(lambda: dict(
        images=0,
        valid_pixels=0,
        changed_pixels=0,
        improved_pixels=0,
        harmed_pixels=0,
        wrong_to_wrong_pixels=0,
    ))
    directions = defaultdict(lambda: dict(
        pixels=0,
        improved_pixels=0,
        harmed_pixels=0,
        wrong_to_wrong_pixels=0,
    ))
    names = {}
    for record in load_jsonl(inputs):
        ctx = record.get('cross_image_bank_recomposition') or {}
        dataset = record.get('dataset_name') or 'unknown'
        space = record.get('cross_image_bank_apply_space') or ctx.get(
            'space') or ''
        variant = record.get('cross_image_bank_apply_variant') or ctx.get(
            'variant') or ''
        min_margin = str(record.get('cross_image_bank_apply_min_bank_margin'))
        max_gap = str(record.get('cross_image_bank_apply_max_base_gap'))
        exp_key = (dataset, space, variant, min_margin, max_gap)
        bucket = overall[exp_key]
        bucket['images'] += 1
        bucket['valid_pixels'] += to_int(ctx.get('valid_pixels'))
        bucket['changed_pixels'] += to_int(ctx.get('final_changed_pixels'))
        bucket['improved_pixels'] += to_int(ctx.get('improved_pixels'))
        bucket['harmed_pixels'] += to_int(ctx.get('harmed_pixels'))
        bucket['wrong_to_wrong_pixels'] += to_int(
            ctx.get('wrong_to_wrong_pixels'))
        for pair in ctx.get('changed_pairs') or []:
            from_idx = to_int(pair.get('from_class_index'))
            to_idx = to_int(pair.get('to_class_index'))
            key = exp_key + (from_idx, to_idx)
            directions[key]['pixels'] += to_int(pair.get('pixels'))
            directions[key]['improved_pixels'] += to_int(
                pair.get('improved_pixels'))
            directions[key]['harmed_pixels'] += to_int(
                pair.get('harmed_pixels'))
            directions[key]['wrong_to_wrong_pixels'] += to_int(
                pair.get('wrong_to_wrong_pixels'))
            names[(dataset, from_idx)] = pair.get('from_class_name', '')
            names[(dataset, to_idx)] = pair.get('to_class_name', '')
    return overall, directions, names


def build_feature_rows(overall, directions, names, pair_summary,
                       class_coverage, min_pixels):
    rows = []
    for key, bucket in sorted(directions.items()):
        (
            dataset, space, variant, min_margin, max_gap,
            from_idx, to_idx,
        ) = key
        pixels = int(bucket['pixels'])
        if pixels < min_pixels:
            continue
        exp_key = (dataset, space, variant, min_margin, max_gap)
        valid_pixels = overall[exp_key]['valid_pixels']
        changed_pixels = overall[exp_key]['changed_pixels']
        from_name = names.get((dataset, from_idx), str(from_idx))
        to_name = names.get((dataset, to_idx), str(to_idx))
        from_role = class_role(from_name)
        to_role = class_role(to_name)

        support = pair_summary.get(
            pair_key(dataset, space, variant, to_idx, from_idx), {})
        opposite = pair_summary.get(
            pair_key(dataset, space, variant, from_idx, to_idx), {})
        from_cov = class_coverage.get((dataset, space, from_idx), {})
        to_cov = class_coverage.get((dataset, space, to_idx), {})

        improved = int(bucket['improved_pixels'])
        harmed = int(bucket['harmed_pixels'])
        wrong_to_wrong = int(bucket['wrong_to_wrong_pixels'])
        net = improved - harmed
        support_margin = to_float(support.get('margin'))
        opposite_margin = to_float(opposite.get('margin'))
        support_pos = to_float(support.get('pos'))
        opposite_pos = to_float(opposite.get('pos'))
        support_pixels = to_int(support.get('pixels'))
        opposite_pixels = to_int(opposite.get('pixels'))
        to_seed_pixels = to_int(to_cov.get('seed_pixels'))
        from_seed_pixels = to_int(from_cov.get('seed_pixels'))
        to_seed_coverage = to_float(to_cov.get('image_seed_coverage'))
        from_seed_coverage = to_float(from_cov.get('image_seed_coverage'))
        row = dict(
            dataset=dataset,
            space=space,
            variant=variant,
            min_bank_margin=min_margin,
            max_base_gap=max_gap,
            from_class_index=from_idx,
            from_class_name=from_name,
            from_role=from_role,
            to_class_index=to_idx,
            to_class_name=to_name,
            to_role=to_role,
            role_pair=f'{from_role}->{to_role}',
            same_role=int(from_role == to_role),
            touches_catch_all=int(
                from_role == 'catch_all' or to_role == 'catch_all'),
            pixels=pixels,
            changed_share=safe_div(pixels, changed_pixels),
            valid_share=safe_div(pixels, valid_pixels),
            improved_pixels=improved,
            harmed_pixels=harmed,
            wrong_to_wrong_pixels=wrong_to_wrong,
            net_pixels=net,
            positive_direction=int(net > 0),
            correction_precision=safe_div(improved, pixels),
            harm_rate=safe_div(harmed, pixels),
            support_margin=support_margin,
            support_positive_ratio=support_pos,
            support_pair_pixels=support_pixels,
            support_pair_images=to_int(support.get('images')),
            support_baseline_gap=to_float(support.get('base_gap')),
            opposite_margin=opposite_margin,
            opposite_positive_ratio=opposite_pos,
            opposite_pair_pixels=opposite_pixels,
            opposite_pair_images=to_int(opposite.get('images')),
            opposite_baseline_gap=to_float(opposite.get('base_gap')),
            asymmetry_margin=support_margin - opposite_margin,
            asymmetry_positive_ratio=support_pos - opposite_pos,
            log_support_pixels=log1p(support_pixels),
            log_opposite_pixels=log1p(opposite_pixels),
            log_support_over_opposite_pixels=math.log(
                (support_pixels + 1.0) / (opposite_pixels + 1.0)),
            from_seed_coverage=from_seed_coverage,
            to_seed_coverage=to_seed_coverage,
            from_seed_pixels=from_seed_pixels,
            to_seed_pixels=to_seed_pixels,
            log_to_seed_pixels=log1p(to_seed_pixels),
            log_from_seed_pixels=log1p(from_seed_pixels),
            log_to_over_from_seed_pixels=math.log(
                (to_seed_pixels + 1.0) / (from_seed_pixels + 1.0)),
            # Audit-only GT columns. Do not use in no-GT rules.
            audit_from_seed_purity=to_float(
                from_cov.get('audit_seed_purity')),
            audit_to_seed_purity=to_float(to_cov.get('audit_seed_purity')),
        )
        rows.append(row)
    return rows


def weighted_auc(rows, score_key):
    pairs = []
    pos_w = 0.0
    neg_w = 0.0
    for row in rows:
        score = to_float(row.get(score_key))
        weight = max(0.0, to_float(row.get('pixels')))
        label = int(row.get('positive_direction'))
        pairs.append((score, label, weight))
        if label:
            pos_w += weight
        else:
            neg_w += weight
    if pos_w <= 0 or neg_w <= 0:
        return None
    win = 0.0
    for i, (score_i, label_i, weight_i) in enumerate(pairs):
        if not label_i:
            continue
        for score_j, label_j, weight_j in pairs:
            if label_j:
                continue
            if score_i > score_j:
                win += weight_i * weight_j
            elif score_i == score_j:
                win += 0.5 * weight_i * weight_j
    return win / (pos_w * neg_w)


def weighted_ap(rows, score_key):
    items = []
    total_pos = 0.0
    for row in rows:
        weight = max(0.0, to_float(row.get('pixels')))
        label = int(row.get('positive_direction'))
        items.append((to_float(row.get(score_key)), label, weight))
        if label:
            total_pos += weight
    if total_pos <= 0:
        return None
    items.sort(key=lambda item: item[0], reverse=True)
    tp = 0.0
    fp = 0.0
    ap = 0.0
    for _, label, weight in items:
        if label:
            tp += weight
            ap += safe_div(tp, tp + fp) * weight
        else:
            fp += weight
    return ap / total_pos


def evaluate_rule(rows, rule):
    selected = [row for row in rows if rule(row)]
    pixels = sum(to_int(row['pixels']) for row in selected)
    improved = sum(to_int(row['improved_pixels']) for row in selected)
    harmed = sum(to_int(row['harmed_pixels']) for row in selected)
    return dict(
        selected_directions=len(selected),
        selected_pixels=pixels,
        improved_pixels=improved,
        harmed_pixels=harmed,
        net_pixels=improved - harmed,
        correction_precision=safe_div(improved, pixels),
        harm_rate=safe_div(harmed, pixels),
    )


def make_rules():
    margins = [-0.10, 0.0, 0.05, 0.10, 0.20, 0.40, 0.60]
    positives = [0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
    asym_margins = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80]
    asym_positives = [0.0, 0.05, 0.10, 0.20, 0.30]
    coverages = [0.0, 0.10, 0.25, 0.40, 0.60, 0.80]
    opposite_max = [0.20, 0.30, 0.40, 0.50, 0.60]
    rules = []
    for margin in margins:
        rules.append((
            f'support_margin>={margin}',
            lambda row, margin=margin: (
                to_float(row['support_margin']) >= margin)))
    for pos in positives:
        rules.append((
            f'support_pos>={pos}',
            lambda row, pos=pos: (
                to_float(row['support_positive_ratio']) >= pos)))
    for margin in margins:
        for pos in positives:
            rules.append((
                f'support_margin>={margin}&support_pos>={pos}',
                lambda row, margin=margin, pos=pos: (
                    to_float(row['support_margin']) >= margin
                    and to_float(row['support_positive_ratio']) >= pos)))
    for margin in margins:
        for pos in positives:
            for asym in asym_margins:
                rules.append((
                    f'margin>={margin}&pos>={pos}&asym_margin>={asym}',
                    lambda row, margin=margin, pos=pos, asym=asym: (
                        to_float(row['support_margin']) >= margin
                        and to_float(row['support_positive_ratio']) >= pos
                        and to_float(row['asymmetry_margin']) >= asym)))
    for pos in positives:
        for asym_pos in asym_positives:
            for opp in opposite_max:
                rules.append((
                    f'pos>={pos}&asym_pos>={asym_pos}&opp_pos<={opp}',
                    lambda row, pos=pos, asym_pos=asym_pos, opp=opp: (
                        to_float(row['support_positive_ratio']) >= pos
                        and to_float(row['asymmetry_positive_ratio'])
                        >= asym_pos
                        and to_float(row['opposite_positive_ratio'])
                        <= opp)))
    for margin in margins:
        for pos in positives:
            for cov in coverages:
                rules.append((
                    f'margin>={margin}&pos>={pos}&to_cov>={cov}&no_catchall',
                    lambda row, margin=margin, pos=pos, cov=cov: (
                        to_float(row['support_margin']) >= margin
                        and to_float(row['support_positive_ratio']) >= pos
                        and to_float(row['to_seed_coverage']) >= cov
                        and int(row['touches_catch_all']) == 0)))
    # Role-generic rule families. They are not dataset-specific pairs.
    safe_to_roles = {
        'built', 'vehicle', 'impervious_surface', 'water',
        'woody_vegetation',
    }
    for margin in margins:
        for pos in positives:
            rules.append((
                f'margin>={margin}&pos>={pos}&to_safe_role&no_catchall',
                lambda row, margin=margin, pos=pos: (
                    to_float(row['support_margin']) >= margin
                    and to_float(row['support_positive_ratio']) >= pos
                    and row['to_role'] in safe_to_roles
                    and int(row['touches_catch_all']) == 0)))
    return rules


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def main():
    parser = argparse.ArgumentParser(
        description='Analyze whether clean-bank correction directions are '
                    'separable by no-GT features.')
    parser.add_argument('--recompose-inputs', nargs='+', required=True)
    parser.add_argument('--pair-summary', required=True)
    parser.add_argument('--class-coverage', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--min-pixels', type=int, default=1000)
    parser.add_argument('--min-train-selected-pixels', type=int, default=10000)
    parser.add_argument('--min-train-precision', type=float, default=0.0)
    args = parser.parse_args()

    pair_summary = load_pair_summary(args.pair_summary)
    class_coverage = load_class_coverage(args.class_coverage)
    overall, directions, names = aggregate_recompose(args.recompose_inputs)
    rows = build_feature_rows(
        overall,
        directions,
        names,
        pair_summary,
        class_coverage,
        args.min_pixels,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    feature_fields = [
        'dataset', 'space', 'variant', 'min_bank_margin', 'max_base_gap',
        'from_class_index', 'from_class_name', 'from_role',
        'to_class_index', 'to_class_name', 'to_role', 'role_pair',
        'same_role', 'touches_catch_all', 'pixels', 'changed_share',
        'valid_share', 'improved_pixels', 'harmed_pixels',
        'wrong_to_wrong_pixels', 'net_pixels', 'positive_direction',
        'correction_precision', 'harm_rate', 'support_margin',
        'support_positive_ratio', 'support_pair_pixels',
        'support_pair_images', 'support_baseline_gap', 'opposite_margin',
        'opposite_positive_ratio', 'opposite_pair_pixels',
        'opposite_pair_images', 'opposite_baseline_gap',
        'asymmetry_margin', 'asymmetry_positive_ratio',
        'log_support_pixels', 'log_opposite_pixels',
        'log_support_over_opposite_pixels', 'from_seed_coverage',
        'to_seed_coverage', 'from_seed_pixels', 'to_seed_pixels',
        'log_to_seed_pixels', 'log_from_seed_pixels',
        'log_to_over_from_seed_pixels', 'audit_from_seed_purity',
        'audit_to_seed_purity',
    ]
    write_csv(
        os.path.join(args.output_dir, 'direction_safety_features.csv'),
        rows,
        feature_fields,
    )

    score_features = [
        'support_margin',
        'support_positive_ratio',
        'asymmetry_margin',
        'asymmetry_positive_ratio',
        'log_support_pixels',
        'log_support_over_opposite_pixels',
        'to_seed_coverage',
        'log_to_seed_pixels',
        'log_to_over_from_seed_pixels',
        'opposite_margin',
        'opposite_positive_ratio',
        'changed_share',
    ]
    feature_rows = []
    groups = [('all', rows)]
    for key in sorted({(r['space'], r['variant']) for r in rows}):
        group_rows = [
            row for row in rows
            if (row['space'], row['variant']) == key
        ]
        groups.append((f'{key[0]}:{key[1]}', group_rows))
    for group_name, group_rows in groups:
        for feature in score_features:
            auc = weighted_auc(group_rows, feature)
            ap = weighted_ap(group_rows, feature)
            neg_auc = weighted_auc(
                [
                    dict(row, **{f'neg_{feature}': -to_float(row[feature])})
                    for row in group_rows
                ],
                f'neg_{feature}',
            )
            neg_ap = weighted_ap(
                [
                    dict(row, **{f'neg_{feature}': -to_float(row[feature])})
                    for row in group_rows
                ],
                f'neg_{feature}',
            )
            feature_rows.append(dict(
                group=group_name,
                feature=feature,
                weighted_auc=auc if auc is not None else '',
                weighted_ap=ap if ap is not None else '',
                neg_weighted_auc=neg_auc if neg_auc is not None else '',
                neg_weighted_ap=neg_ap if neg_ap is not None else '',
                rows=len(group_rows),
                positive_rows=sum(
                    1 for row in group_rows
                    if int(row['positive_direction'])),
                pixels=sum(to_int(row['pixels']) for row in group_rows),
                positive_pixels=sum(
                    to_int(row['pixels']) for row in group_rows
                    if int(row['positive_direction'])),
            ))
    write_csv(
        os.path.join(args.output_dir, 'direction_safety_feature_auc.csv'),
        feature_rows,
        [
            'group', 'feature', 'weighted_auc', 'weighted_ap',
            'neg_weighted_auc', 'neg_weighted_ap', 'rows',
            'positive_rows', 'pixels', 'positive_pixels',
        ],
    )

    rule_defs = make_rules()
    rule_rows = []
    for group_name, group_rows in groups:
        for rule_name, rule in rule_defs:
            stats = evaluate_rule(group_rows, rule)
            rule_rows.append(dict(group=group_name, rule=rule_name, **stats))
    write_csv(
        os.path.join(args.output_dir, 'direction_safety_rule_summary.csv'),
        rule_rows,
        [
            'group', 'rule', 'selected_directions', 'selected_pixels',
            'improved_pixels', 'harmed_pixels', 'net_pixels',
            'correction_precision', 'harm_rate',
        ],
    )

    lodo_rows = []
    datasets = sorted({row['dataset'] for row in rows})
    exp_groups = sorted({(row['space'], row['variant']) for row in rows})
    for exp_group in exp_groups:
        group_rows = [
            row for row in rows
            if (row['space'], row['variant']) == exp_group
        ]
        for heldout in datasets:
            train_rows = [
                row for row in group_rows if row['dataset'] != heldout]
            test_rows = [
                row for row in group_rows if row['dataset'] == heldout]
            if not train_rows or not test_rows:
                continue
            best = None
            best_rule_name = None
            best_rule = None
            for rule_name, rule in rule_defs:
                train_stats = evaluate_rule(train_rows, rule)
                if (
                        train_stats['selected_pixels']
                        < args.min_train_selected_pixels):
                    continue
                if (
                        train_stats['correction_precision']
                        < args.min_train_precision):
                    continue
                if best is None or (
                        train_stats['net_pixels'],
                        train_stats['correction_precision'],
                        train_stats['selected_pixels'],
                ) > (
                        best['net_pixels'],
                        best['correction_precision'],
                        best['selected_pixels'],
                ):
                    best = train_stats
                    best_rule_name = rule_name
                    best_rule = rule
            if best is None:
                continue
            test_stats = evaluate_rule(test_rows, best_rule)
            oracle_stats = evaluate_rule(
                test_rows,
                lambda row: int(row['positive_direction']) == 1,
            )
            lodo_rows.append(dict(
                space=exp_group[0],
                variant=exp_group[1],
                heldout_dataset=heldout,
                selected_rule=best_rule_name,
                train_selected_directions=best['selected_directions'],
                train_selected_pixels=best['selected_pixels'],
                train_net_pixels=best['net_pixels'],
                train_precision=best['correction_precision'],
                test_selected_directions=test_stats['selected_directions'],
                test_selected_pixels=test_stats['selected_pixels'],
                test_improved_pixels=test_stats['improved_pixels'],
                test_harmed_pixels=test_stats['harmed_pixels'],
                test_net_pixels=test_stats['net_pixels'],
                test_precision=test_stats['correction_precision'],
                test_harm_rate=test_stats['harm_rate'],
                heldout_oracle_positive_net=oracle_stats['net_pixels'],
                heldout_oracle_positive_pixels=oracle_stats[
                    'selected_pixels'],
            ))
    write_csv(
        os.path.join(args.output_dir, 'direction_safety_lodo_summary.csv'),
        lodo_rows,
        [
            'space', 'variant', 'heldout_dataset', 'selected_rule',
            'train_selected_directions', 'train_selected_pixels',
            'train_net_pixels', 'train_precision',
            'test_selected_directions', 'test_selected_pixels',
            'test_improved_pixels', 'test_harmed_pixels',
            'test_net_pixels', 'test_precision', 'test_harm_rate',
            'heldout_oracle_positive_net', 'heldout_oracle_positive_pixels',
        ],
    )

    print(f'Rows: {len(rows)}')
    print(f'Wrote {args.output_dir}')


if __name__ == '__main__':
    main()
