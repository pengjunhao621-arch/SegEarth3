import argparse
import csv
import glob
import math
import os


DEFAULT_POSITIVE_PAIRS = (
    'vdd:roof->facade,'
    'potsdam:tree->grass,'
    'vaihingen:grass->tree,'
    'openearthmap:pavement->building'
)

DEFAULT_NEGATIVE_PAIRS = (
    'vdd:facade->roof,'
    'openearthmap:grass->tree,'
    'udd5:road->background,'
    'udd5:vegetation->background,'
    'udd5:building->background,'
    'loveda:agricultural->background,'
    'loveda:forest->background'
)

BACKGROUND_NAMES = {
    'background', 'clutter', 'other', 'others', 'ignore', 'unknown',
}

VEGETATION_NAMES = {
    'tree', 'trees', 'grass', 'forest', 'agricultural', 'agriculture',
    'cropland', 'vegetation', 'low vegetation', 'low_vegetation',
}

BUILT_SURFACE_NAMES = {
    'building', 'buildings', 'roof', 'facade', 'road', 'pavement',
    'impervious surface', 'impervious_surface', 'bareland', 'bare land',
}

OBJECT_NAMES = {
    'car', 'vehicle', 'vehicles', 'ship', 'airplane', 'plane',
}

WATER_NAMES = {'water', 'river', 'lake', 'sea'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build no-GT selector feature tables from evidence-bias summaries.')
    parser.add_argument(
        '--summary-dirs',
        nargs='+',
        required=True,
        help='Evidence-bias summary dirs, e.g. logs/evidence_bias_scan/*/summary.')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Output directory for selector feature CSV files.')
    parser.add_argument(
        '--support-threshold',
        default='0.1',
        help='Support threshold row to merge from bias_support_summary.csv.')
    parser.add_argument(
        '--effect-csvs',
        nargs='*',
        default=[],
        help='Optional scale/expert rerank pair summary CSVs used only as labels/effects.')
    parser.add_argument(
        '--positive-pairs',
        default=DEFAULT_POSITIVE_PAIRS,
        help='Comma/semicolon separated dataset:target->competitor pairs known positive.')
    parser.add_argument(
        '--negative-pairs',
        default=DEFAULT_NEGATIVE_PAIRS,
        help='Comma/semicolon separated dataset:target->competitor pairs known risky/negative.')
    parser.add_argument(
        '--min-pixels',
        type=int,
        default=1,
        help='Ignore pair directions with fewer pair pixels in bias summaries.')
    parser.add_argument(
        '--hard-win-threshold',
        type=float,
        default=0.25,
        help='Best-probe target-win ratio below this is marked hard competition.')
    parser.add_argument(
        '--reject-base-win-threshold',
        type=float,
        default=0.70,
        help='Base target-win ratio above this is marked threshold/reject-like.')
    return parser.parse_args()


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(matched)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(set(paths))


def read_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def as_float(value, default=0.0):
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    return int(round(as_float(value, default)))


def norm_name(name):
    return str(name or '').strip().lower()


def pair_key(dataset, target, competitor):
    return (norm_name(dataset), norm_name(target), norm_name(competitor))


def parse_pair_set(text):
    pairs = set()
    if not text:
        return pairs
    clean = text.replace(';', ',')
    for item in clean.split(','):
        item = item.strip()
        if not item or ':' not in item or '->' not in item:
            continue
        dataset, rest = item.split(':', 1)
        target, competitor = rest.split('->', 1)
        pairs.add(pair_key(dataset, target, competitor))
    return pairs


def class_role(name):
    key = norm_name(name)
    if key in BACKGROUND_NAMES:
        return 'background'
    if key in VEGETATION_NAMES:
        return 'vegetation'
    if key in BUILT_SURFACE_NAMES:
        return 'built_surface'
    if key in OBJECT_NAMES:
        return 'object'
    if key in WATER_NAMES:
        return 'water'
    return 'other'


def pair_role(target, competitor):
    target_role = class_role(target)
    competitor_role = class_role(competitor)
    if 'background' in (target_role, competitor_role):
        return 'foreground_background'
    if target_role == competitor_role:
        return f'{target_role}_intra'
    return f'{target_role}_vs_{competitor_role}'


def probe_family(probe):
    probe = norm_name(probe)
    if probe.startswith('scale'):
        return 'scale'
    if probe.startswith('blur'):
        return 'low_frequency'
    if probe.startswith('sharpen'):
        return 'high_frequency'
    if probe.startswith('hflip') or probe.startswith('vflip') or probe.startswith('rot'):
        return 'orientation'
    if probe.startswith('pad'):
        return 'context_position'
    return 'other'


def choose_best(rows, key):
    if not rows:
        return None
    return max(rows, key=lambda row: (as_float(row.get(key)), as_float(row.get('pair_pixels'))))


def load_bias_summaries(summary_dirs, support_threshold, min_pixels):
    pair_rows = {}
    probe_rows = []
    support_rows = {}
    for summary_dir in expand_paths(summary_dirs):
        pair_path = os.path.join(summary_dir, 'bias_pair_summary.csv')
        support_path = os.path.join(summary_dir, 'bias_support_summary.csv')
        for row in read_csv(pair_path):
            if row.get('head') != 'final':
                continue
            pixels = as_int(row.get('pair_pixels'))
            if pixels < min_pixels:
                continue
            key = pair_key(
                row.get('dataset_name'),
                row.get('target_class_name'),
                row.get('competitor_class_name'))
            row = dict(row)
            row['_pair_key'] = key
            probe_rows.append(row)
            pair_rows.setdefault(key, []).append(row)
        for row in read_csv(support_path):
            if row.get('head') != 'final':
                continue
            if str(row.get('support_threshold')) != str(support_threshold):
                continue
            key = pair_key(
                row.get('dataset_name'),
                row.get('target_class_name'),
                row.get('competitor_class_name'))
            row = dict(row)
            row['_pair_key'] = key
            support_rows.setdefault(key, []).append(row)
    return pair_rows, probe_rows, support_rows


def load_effect_rows(effect_csvs):
    effects = {}
    for path in expand_paths(effect_csvs):
        for row in read_csv(path):
            key = pair_key(
                row.get('dataset_name'),
                row.get('target_class_name'),
                row.get('competitor_class_name'))
            gate = as_int(row.get('gate_valid_pixels') or row.get('gate_pixels'))
            changed = as_int(row.get('changed_gate_pixels'))
            improved = as_int(row.get('improved_gate_pixels'))
            harmed = as_int(row.get('harmed_gate_pixels'))
            real_net = as_int(row.get('net_improved_gate_pixels'))
            cf_net = as_int(row.get('gate_counterfactual_net_pixels'))
            cf_improved = as_int(row.get('gate_counterfactual_improved_pixels'))
            cf_harmed = as_int(row.get('gate_counterfactual_harmed_pixels'))
            if cf_improved == 0 and cf_harmed == 0:
                cf_improved = as_int(row.get('gate_true_target_pixels'))
                cf_harmed = as_int(row.get('gate_true_competitor_pixels'))
                cf_net = cf_improved - cf_harmed
            denom = cf_improved + cf_harmed
            candidate = {
                'effect_source': path,
                'effect_gate_pixels': gate,
                'effect_changed_pixels': changed,
                'effect_improved_pixels': improved,
                'effect_harmed_pixels': harmed,
                'effect_real_net_pixels': real_net,
                'effect_real_precision': (
                    '' if improved + harmed == 0 else improved / (improved + harmed)),
                'effect_counterfactual_net_pixels': cf_net,
                'effect_counterfactual_precision': (
                    '' if denom == 0 else cf_improved / denom),
            }
            previous = effects.get(key)
            if previous is None:
                effects[key] = candidate
                continue
            prev_score = (
                abs(as_int(previous.get('effect_real_net_pixels')))
                + abs(as_int(previous.get('effect_counterfactual_net_pixels'))))
            cand_score = abs(real_net) + abs(cf_net)
            if cand_score > prev_score:
                effects[key] = candidate
    return effects


def get_probe_value(rows, probe_names, field):
    for name in probe_names:
        for row in rows:
            if row.get('probe_name') == name:
                return as_float(row.get(field))
    return ''


def safe_ratio(num, den):
    return '' if den == 0 else num / den


def build_pair_features(pair_rows, support_rows, effects, positive_pairs,
                        negative_pairs, args):
    rows = []
    probe_detail = []
    for key, probes in pair_rows.items():
        probes = sorted(probes, key=lambda r: r.get('probe_name') or '')
        best = choose_best(probes, 'mean_relative_stability')
        if best is None:
            continue
        support = support_rows.get(key, [])
        best_support = None
        if support:
            same_probe = [r for r in support if r.get('probe_name') == best.get('probe_name')]
            best_support = same_probe[0] if same_probe else choose_best(
                support, 'mean_relative_stability')

        dataset, target, competitor = key
        reverse_key = (dataset, competitor, target)
        reverse_best = choose_best(pair_rows.get(reverse_key, []),
                                   'mean_relative_stability')

        pixels = as_int(best.get('pair_pixels'))
        reverse_pixels = as_int(reverse_best.get('pair_pixels')) if reverse_best else 0
        base_win = as_float(best.get('base_target_wins_ratio'))
        probe_win = as_float(best.get('probe_target_wins_ratio'))
        best_rel = as_float(best.get('mean_relative_stability'))
        reverse_rel = (
            as_float(reverse_best.get('mean_relative_stability'))
            if reverse_best else '')
        reverse_win = (
            as_float(reverse_best.get('probe_target_wins_ratio'))
            if reverse_best else '')
        target_only = (
            as_float(best_support.get('probe_target_only_support_ratio'))
            if best_support else '')
        competitor_only = (
            as_float(best_support.get('probe_competitor_only_support_ratio'))
            if best_support else '')
        target_gain = (
            as_float(best_support.get('target_gain_support_ratio'))
            if best_support else '')
        competitor_lost = (
            as_float(best_support.get('competitor_lost_support_ratio'))
            if best_support else '')

        is_threshold_like = base_win >= args.reject_base_win_threshold
        is_hard = (
            base_win < args.reject_base_win_threshold
            and probe_win < args.hard_win_threshold)
        is_recoverable = (
            base_win < args.reject_base_win_threshold
            and probe_win >= args.hard_win_threshold)
        if is_threshold_like:
            regime = 'threshold_or_reject_like'
        elif is_hard:
            regime = 'hard_competition'
        elif is_recoverable:
            regime = 'probe_recoverable'
        else:
            regime = 'uncertain'

        label = 'unknown'
        if key in positive_pairs:
            label = 'known_positive'
        if key in negative_pairs:
            label = 'known_negative'
        effect = effects.get(key, {})
        real_net = as_int(effect.get('effect_real_net_pixels'))
        cf_net = as_int(effect.get('effect_counterfactual_net_pixels'))
        if label == 'unknown' and (real_net or cf_net):
            if real_net > 0 or (real_net == 0 and cf_net > 0):
                label = 'effect_positive'
            elif real_net < 0 or (real_net == 0 and cf_net < 0):
                label = 'effect_negative'

        row = {
            'dataset_name': dataset,
            'target_class_name': target,
            'competitor_class_name': competitor,
            'selector_label': label,
            'pair_role': pair_role(target, competitor),
            'target_role': class_role(target),
            'competitor_role': class_role(competitor),
            'selector_regime': regime,
            'pair_pixels': pixels,
            'reverse_pair_pixels': reverse_pixels,
            'log_pixel_ratio_vs_reverse': (
                '' if reverse_pixels <= 0 else math.log((pixels + 1) / (reverse_pixels + 1))),
            'best_probe_name': best.get('probe_name'),
            'best_probe_family': probe_family(best.get('probe_name')),
            'best_relative_stability': best_rel,
            'best_margin_gain': as_float(best.get('mean_margin_gain')),
            'best_target_drop': as_float(best.get('mean_target_drop')),
            'best_competitor_drop': as_float(best.get('mean_competitor_drop')),
            'base_target_wins_ratio': base_win,
            'probe_target_wins_ratio': probe_win,
            'target_win_gain': probe_win - base_win,
            'base_margin': as_float(best.get('mean_base_direct_margin')),
            'probe_margin': as_float(best.get('mean_probe_margin')),
            'reverse_best_probe_name': (
                reverse_best.get('probe_name') if reverse_best else ''),
            'reverse_best_relative_stability': reverse_rel,
            'reverse_probe_target_wins_ratio': reverse_win,
            'asymmetry_relative_stability': (
                '' if reverse_rel == '' else best_rel - reverse_rel),
            'asymmetry_probe_win': (
                '' if reverse_win == '' else probe_win - reverse_win),
            'support_threshold': args.support_threshold,
            'probe_target_support_ratio': (
                as_float(best_support.get('probe_target_support_ratio'))
                if best_support else ''),
            'probe_competitor_support_ratio': (
                as_float(best_support.get('probe_competitor_support_ratio'))
                if best_support else ''),
            'probe_target_only_support_ratio': target_only,
            'probe_competitor_only_support_ratio': competitor_only,
            'target_gain_support_ratio': target_gain,
            'competitor_lost_support_ratio': competitor_lost,
            'target_creation_score': (
                '' if target_only == '' else target_only + target_gain),
            'competitor_suppression_score': (
                '' if competitor_only == '' else competitor_only + competitor_lost),
            'scale05_relative_stability': get_probe_value(
                probes, ['scale:0.5', 'scale:0.50'], 'mean_relative_stability'),
            'scale075_relative_stability': get_probe_value(
                probes, ['scale:0.75'], 'mean_relative_stability'),
            'blur_relative_stability': get_probe_value(
                probes, ['blur:1.5'], 'mean_relative_stability'),
            'sharpen_relative_stability': get_probe_value(
                probes, ['sharpen'], 'mean_relative_stability'),
            'hflip_relative_stability': get_probe_value(
                probes, ['hflip'], 'mean_relative_stability'),
            'rot90_relative_stability': get_probe_value(
                probes, ['rot90'], 'mean_relative_stability'),
            'pad_relative_stability': get_probe_value(
                probes, ['pad:128'], 'mean_relative_stability'),
            'frequency_contrast_blur_minus_sharpen': (
                as_float(get_probe_value(probes, ['blur:1.5'], 'mean_relative_stability'))
                - as_float(get_probe_value(probes, ['sharpen'], 'mean_relative_stability'))),
            'scale_vs_frequency_max': max(
                as_float(get_probe_value(probes, ['scale:0.5', 'scale:0.50'],
                                         'mean_relative_stability')),
                as_float(get_probe_value(probes, ['scale:0.75'],
                                         'mean_relative_stability')),
            ) - max(
                as_float(get_probe_value(probes, ['blur:1.5'],
                                         'mean_relative_stability')),
                as_float(get_probe_value(probes, ['sharpen'],
                                         'mean_relative_stability')),
            ),
        }
        row.update(effect)
        rows.append(row)

        for probe in probes:
            detail = {
                'dataset_name': dataset,
                'target_class_name': target,
                'competitor_class_name': competitor,
                'selector_label': label,
                'pair_role': pair_role(target, competitor),
                'probe_name': probe.get('probe_name'),
                'probe_family': probe_family(probe.get('probe_name')),
                'pair_pixels': as_int(probe.get('pair_pixels')),
                'mean_relative_stability': as_float(
                    probe.get('mean_relative_stability')),
                'mean_margin_gain': as_float(probe.get('mean_margin_gain')),
                'base_target_wins_ratio': as_float(
                    probe.get('base_target_wins_ratio')),
                'probe_target_wins_ratio': as_float(
                    probe.get('probe_target_wins_ratio')),
                'mean_target_drop': as_float(probe.get('mean_target_drop')),
                'mean_competitor_drop': as_float(
                    probe.get('mean_competitor_drop')),
            }
            probe_detail.append(detail)
    return rows, probe_detail


def build_dataset_summary(rows):
    buckets = {}
    for row in rows:
        key = (row.get('dataset_name'), row.get('selector_regime'))
        bucket = buckets.setdefault(key, {
            'dataset_name': row.get('dataset_name'),
            'selector_regime': row.get('selector_regime'),
            'pair_count': 0,
            'pair_pixels': 0,
            'known_positive_count': 0,
            'known_negative_count': 0,
            'effect_positive_count': 0,
            'effect_negative_count': 0,
        })
        bucket['pair_count'] += 1
        bucket['pair_pixels'] += as_int(row.get('pair_pixels'))
        label = row.get('selector_label')
        if label == 'known_positive':
            bucket['known_positive_count'] += 1
        elif label == 'known_negative':
            bucket['known_negative_count'] += 1
        elif label == 'effect_positive':
            bucket['effect_positive_count'] += 1
        elif label == 'effect_negative':
            bucket['effect_negative_count'] += 1
    totals = {}
    for row in rows:
        totals[row.get('dataset_name')] = (
            totals.get(row.get('dataset_name'), 0) + as_int(row.get('pair_pixels')))
    out = []
    for bucket in buckets.values():
        total = totals.get(bucket['dataset_name'], 0)
        bucket['pixel_ratio_in_dataset_scan'] = safe_ratio(
            bucket['pair_pixels'], total)
        out.append(bucket)
    out.sort(key=lambda r: (r['dataset_name'], r['selector_regime']))
    return out


def write_csv(path, rows, preferred):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for key in preferred:
        if key not in keys:
            keys.append(key)
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    positive_pairs = parse_pair_set(args.positive_pairs)
    negative_pairs = parse_pair_set(args.negative_pairs)
    pair_rows, probe_rows, support_rows = load_bias_summaries(
        args.summary_dirs, args.support_threshold, args.min_pixels)
    if not pair_rows:
        raise FileNotFoundError('No final-head bias_pair_summary rows found.')
    effects = load_effect_rows(args.effect_csvs)
    selector_rows, probe_detail = build_pair_features(
        pair_rows, support_rows, effects, positive_pairs, negative_pairs, args)
    selector_rows.sort(key=lambda row: (
        row.get('dataset_name'),
        row.get('selector_regime'),
        -as_float(row.get('pair_pixels')),
        -as_float(row.get('best_relative_stability')),
    ))
    probe_detail.sort(key=lambda row: (
        row.get('dataset_name'),
        row.get('target_class_name'),
        row.get('competitor_class_name'),
        -as_float(row.get('mean_relative_stability')),
    ))
    dataset_rows = build_dataset_summary(selector_rows)

    pair_preferred = [
        'dataset_name',
        'target_class_name',
        'competitor_class_name',
        'selector_label',
        'pair_role',
        'selector_regime',
        'pair_pixels',
        'best_probe_name',
        'best_probe_family',
        'best_relative_stability',
        'base_target_wins_ratio',
        'probe_target_wins_ratio',
        'target_win_gain',
        'probe_target_only_support_ratio',
        'target_gain_support_ratio',
        'competitor_lost_support_ratio',
        'probe_competitor_only_support_ratio',
        'asymmetry_relative_stability',
        'asymmetry_probe_win',
        'reverse_pair_pixels',
        'reverse_best_probe_name',
        'reverse_best_relative_stability',
        'effect_counterfactual_net_pixels',
        'effect_counterfactual_precision',
        'effect_real_net_pixels',
        'effect_real_precision',
        'effect_source',
    ]
    probe_preferred = [
        'dataset_name',
        'target_class_name',
        'competitor_class_name',
        'selector_label',
        'pair_role',
        'probe_name',
        'probe_family',
        'pair_pixels',
        'mean_relative_stability',
        'mean_margin_gain',
        'base_target_wins_ratio',
        'probe_target_wins_ratio',
        'mean_target_drop',
        'mean_competitor_drop',
    ]
    dataset_preferred = [
        'dataset_name',
        'selector_regime',
        'pair_count',
        'pair_pixels',
        'pixel_ratio_in_dataset_scan',
        'known_positive_count',
        'known_negative_count',
        'effect_positive_count',
        'effect_negative_count',
    ]
    write_csv(
        os.path.join(args.out_dir, 'selector_pair_features.csv'),
        selector_rows,
        pair_preferred)
    write_csv(
        os.path.join(args.out_dir, 'selector_probe_features.csv'),
        probe_detail,
        probe_preferred)
    write_csv(
        os.path.join(args.out_dir, 'selector_dataset_summary.csv'),
        dataset_rows,
        dataset_preferred)


if __name__ == '__main__':
    main()
