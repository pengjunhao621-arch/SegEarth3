"""Contracts for the frozen role-functional SAM3 text screen."""

import json


PROTOCOL = 'role_functional_text_screen_v1'
SCHEMA_VERSION = 1
BANK_SCHEMA_VERSION = 1
PI_PROTOCOL = 'pi_role_compatibility_v1'
PI_SCHEMA_VERSION = 1
COMPLETION_PROTOCOL = 'role_text_completion_v1'
COMPLETION_SCHEMA_VERSION = 1
PE_PROTOCOL = 'pe_role_evidence_v1'
PE_SCHEMA_VERSION = 1
BOUNDARY_REPLAY_PROTOCOL = 'boundary_replay_v1'
BOUNDARY_REPLAY_SCHEMA_VERSION = 1
BOUNDARY_REPLAY_TRACE_TOLERANCE = 5e-3
CLASS_ROLE_ALIGNMENT_PROTOCOL = 'class_role_alignment_v1'
CLASS_ROLE_ALIGNMENT_SCHEMA_VERSION = 1
CLASS_ROLE_ALIGNMENT_PE_LAYER = 18

CRA_COCO_WEIGHTS = (
    ('l030', 0.30),
    ('l050', 0.50),
    ('l070', 0.70),
    ('l090', 0.90),
)
CRA_FCRA_TEMPERATURES = (
    ('t005', 0.05),
    ('t010', 0.10),
)
CRA_VARIANT_NAMES = (
    'cra_official',
    'cra_best_full',
    *(f'cra_{base}_coco_{weight}'
      for base in ('official', 'best')
      for weight, _ in CRA_COCO_WEIGHTS),
    *(f'cra_best_fcra_{evidence}_{temperature}'
      for evidence in ('d', 'gamma')
      for temperature, _ in CRA_FCRA_TEMPERATURES),
)

PE_LAYER_IDS = (7, 15, 23, 31)
PE_FEATURE_NAMES = (
    'block07',
    'block15',
    'block23',
    'block31',
    'fpn_final',
    'rgb',
)
PE_SEMANTIC_SOURCES = (
    'unrefined',
    *PE_FEATURE_NAMES,
)
PE_ADMISSION_MODES = ('native', 'anchor_admission')
PE_VARIANT_NAMES = tuple(
    f'pe_sem_{source}_{admission}'
    for source in PE_SEMANTIC_SOURCES
    for admission in PE_ADMISSION_MODES
)

BOUNDARY_GUIDES = ('block23', 'rgb', 'uniform')
BOUNDARY_STRENGTHS = (
    ('l025', 0.25),
    ('l050', 0.50),
    ('l100', 1.00),
)
BOUNDARY_PRIMARY_STRENGTH = 'l050'
REPLAY_VARIANT_NAMES = (
    'br_official',
    'br_best_full',
    'br_replay_p',
    'br_replay_s',
    'br_replay_i',
    'br_replay_all',
)
BOUNDARY_VARIANT_NAMES = tuple(
    f'br_{base}_{guide}_{strength}'
    for guide in BOUNDARY_GUIDES
    for strength, _ in BOUNDARY_STRENGTHS
    for base in ('official', 'best')
)
BOUNDARY_REPLAY_VARIANT_NAMES = (
    *REPLAY_VARIANT_NAMES,
    *BOUNDARY_VARIANT_NAMES,
)

PI_VARIANT_NAMES = (
    'pi_native_p0_i0',
    'pi_native_p1_i0',
    'pi_native_p0_i1',
    'pi_native_p1_i1',
    'pi_freeze_admission_p1_i1',
    'pi_freeze_amplitude_p1_i1',
    'pi_hold_i_only_winner_p1_i1',
    'pi_branch_once_p0_i0',
    'pi_branch_once_p1_i0',
    'pi_branch_once_p0_i1',
    'pi_branch_once_p1_i1',
)

PI_MECHANISM_MAP_NAMES = (
    'semantic_anchor',
    'instance_i_only',
    'instance_pi',
    'admission_added_support',
    'admission_removed_support',
)

ROLE_FIELDS = (
    ('presence', 'presence_candidates', 2),
    ('semantic', 'semantic_candidates', 3),
    ('instance', 'instance_candidates', 2),
)

# Five deliberately sparse points: alpha sensitivity at a fixed clip and clip
# sensitivity at a fixed alpha.  This is a stability diagnostic, not a per-
# dataset hyper-parameter search.
RESIDUAL_SETTINGS = (
    ('alpha025_clip025', 0.25, 0.25),
    ('alpha050_clip010', 0.50, 0.10),
    ('alpha050_clip025', 0.50, 0.25),
    ('alpha050_clip050', 0.50, 0.50),
    ('alpha100_clip025', 1.00, 0.25),
)
DEFAULT_SETTING = 'alpha050_clip025'


def combo_variant_name(presence_slot, semantic_slot, instance_slot):
    return (
        f'combo_p{int(presence_slot)}_s{int(semantic_slot)}_'
        f'i{int(instance_slot)}')


def sensitivity_variant_name(role, slot, setting):
    return f'sensitivity_{role}_{int(slot)}_{setting}'


def completion_variant_name(presence_slot, semantic_slot, instance_slot):
    return (
        f'anchor_admission_p{int(presence_slot)}_s{int(semantic_slot)}_'
        f'i{int(instance_slot)}')


COMBINATION_VARIANTS = tuple(
    combo_variant_name(presence_slot, semantic_slot, instance_slot)
    for presence_slot in range(3)
    for semantic_slot in range(4)
    for instance_slot in range(3)
)


def load_role_text_selection_registry(path, dataset):
    """Load the frozen per-dataset role compositions used by later screens."""
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    if int(payload.get('schema_version', -1)) != 1:
        raise ValueError(f'{path} must use schema_version=1.')
    datasets = payload.get('datasets')
    if not isinstance(datasets, dict):
        raise ValueError(f'{path} needs a datasets mapping.')
    key = str(dataset).lower()
    if key not in datasets:
        raise ValueError(f'{path} has no selection for {key!r}.')
    record = dict(datasets[key])

    def slots(field):
        value = record.get(field, {}).get('slots')
        if (not isinstance(value, list) or len(value) != 3
                or any(not isinstance(slot, int) for slot in value)):
            raise ValueError(f'{key}.{field}.slots must be [P, S, I].')
        limits = (2, 3, 2)
        value = tuple(int(slot) for slot in value)
        if any(slot < 0 or slot > limit
               for slot, limit in zip(value, limits)):
            raise ValueError(f'{key}.{field}.slots is out of range: {value}.')
        return value

    record['_best_overall_slots'] = slots('best_overall')
    record['_best_all_nonzero_slots'] = slots('best_all_nonzero')
    if not all(slot > 0 for slot in record['_best_all_nonzero_slots']):
        raise ValueError(f'{key}.best_all_nonzero must not contain slot 0.')
    instance_slot = int(record.get('instance_diagnostic_slot', -1))
    if instance_slot not in (1, 2):
        raise ValueError(f'{key}.instance_diagnostic_slot must be 1 or 2.')
    record['_instance_diagnostic_slot'] = instance_slot
    prompt_bank = record.get('prompt_bank')
    if not isinstance(prompt_bank, str) or not prompt_bank.strip():
        raise ValueError(f'{key}.prompt_bank must be a non-empty path.')
    record['_dataset'] = key
    record['_path'] = path
    return record

SENSITIVITY_VARIANTS = tuple(
    sensitivity_variant_name(role, slot, setting)
    for role, _, count in ROLE_FIELDS
    for slot in range(1, count + 1)
    for setting, _, _ in RESIDUAL_SETTINGS
    if setting != DEFAULT_SETTING
)

VARIANT_NAMES = (
    'baseline',
    *COMBINATION_VARIANTS,
    'semantic_head_anchor',
    'semantic_head_s1',
    'semantic_head_s2',
    'semantic_head_s3',
    'instance_head_anchor',
    'instance_head_i1',
    'instance_head_i2',
    'shared_semantic_s1',
    'shared_semantic_s2',
    'shared_semantic_s3',
    *SENSITIVITY_VARIANTS,
)


def reference_variant(name):
    if name.startswith('semantic_head_'):
        return 'semantic_head_anchor'
    if name.startswith('instance_head_'):
        return 'instance_head_anchor'
    return 'baseline'


def _normalise_prompt(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a non-empty string.')
    return ' '.join(value.strip().split())


def load_role_functional_text_bank(
        path, expected_class_names=None, expected_official_prompts=None):
    """Load a frozen bank and verify its exact baseline anchor strings."""
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    if int(payload.get('schema_version', -1)) != BANK_SCHEMA_VERSION:
        raise ValueError(
            f'Prompt bank {path} must use schema_version='
            f'{BANK_SCHEMA_VERSION}.')
    if payload.get('protocol') != PROTOCOL:
        raise ValueError(
            f'Prompt bank {path} must declare protocol={PROTOCOL!r}.')
    if payload.get('frozen_before_evaluation') is not True:
        raise ValueError(
            f'Prompt bank {path} must set frozen_before_evaluation=true.')

    classes = payload.get('classes')
    if not isinstance(classes, list) or not classes:
        raise ValueError(f'Prompt bank {path} needs a non-empty classes list.')
    if [int(item.get('id', -1)) for item in classes] != list(
            range(len(classes))):
        raise ValueError(
            f'Prompt bank {path} class IDs must be contiguous and ordered.')

    observed_names = []
    observed_official = []
    for item in classes:
        name = _normalise_prompt(item.get('name'), 'class name')
        official = item.get('official_prompts')
        if not isinstance(official, list) or not official:
            raise ValueError(f'{name}.official_prompts must be non-empty.')
        official = [
            _normalise_prompt(value, f'{name}.official_prompts')
            for value in official
        ]
        if len({value.lower() for value in official}) != len(official):
            raise ValueError(f'{name}.official_prompts contains duplicates.')

        descriptions = list(official)
        for role, field, expected_count in ROLE_FIELDS:
            values = item.get(field)
            if not isinstance(values, list) or len(values) != expected_count:
                raise ValueError(
                    f'{name}.{field} needs exactly {expected_count} prompts.')
            values = [
                _normalise_prompt(value, f'{name}.{field}')
                for value in values
            ]
            if len({value.lower() for value in values}) != len(values):
                raise ValueError(f'{name}.{field} contains duplicates.')
            item[field] = values
            descriptions.extend(values)

        item['name'] = name
        item['official_prompts'] = official
        item['descriptions'] = list(dict.fromkeys(descriptions))
        observed_names.append(name.lower())
        observed_official.append([value.lower() for value in official])

    if expected_class_names is not None:
        expected = [str(value).strip().lower()
                    for value in expected_class_names]
        if observed_names != expected:
            raise ValueError(
                f'Prompt bank class order mismatch. expected={expected}, '
                f'observed={observed_names}.')
    if expected_official_prompts is not None:
        expected = [
            [_normalise_prompt(value, 'expected official prompt').lower()
             for value in prompts]
            for prompts in expected_official_prompts
        ]
        if observed_official != expected:
            raise ValueError(
                'Prompt bank official_prompts do not exactly match the '
                f'baseline class file. expected={expected}, '
                f'observed={observed_official}.')

    completion = payload.get('completion')
    if completion is not None:
        if not isinstance(completion, dict):
            raise ValueError(f'Prompt bank {path} completion must be a dict.')
        if completion.get('protocol') != COMPLETION_PROTOCOL:
            raise ValueError(
                f'Prompt bank {path} completion must declare protocol='
                f'{COMPLETION_PROTOCOL!r}.')

        def normalize_combination(value, field):
            if (not isinstance(value, list) or len(value) != 3
                    or any(not isinstance(slot, int) for slot in value)):
                raise ValueError(f'{field} must be [P, S, I] integer slots.')
            slots = tuple(int(slot) for slot in value)
            limits = (2, 3, 2)
            if any(slot < 0 or slot > limit
                   for slot, limit in zip(slots, limits)):
                raise ValueError(f'{field} has an out-of-range slot: {slots}.')
            return slots

        selected = normalize_combination(
            completion.get('selected_combination'),
            'completion.selected_combination')
        requested = completion.get('anchor_admission_combinations')
        if not isinstance(requested, list) or not requested:
            raise ValueError(
                'completion.anchor_admission_combinations must be non-empty.')
        combinations = [
            normalize_combination(value, 'anchor_admission_combinations')
            for value in requested
        ]
        if len(set(combinations)) != len(combinations):
            raise ValueError('Anchor-admission combinations contain duplicates.')
        if selected not in combinations:
            raise ValueError(
                'selected_combination must have an anchor-admission control.')
        requested_targets = completion.get('target_combinations', [])
        if not isinstance(requested_targets, list):
            raise ValueError('completion.target_combinations must be a list.')
        targets = [
            normalize_combination(value, 'target_combinations')
            for value in requested_targets
        ]
        if len(set(targets)) != len(targets):
            raise ValueError('Target combinations contain duplicates.')
        if any(target not in combinations for target in targets):
            raise ValueError(
                'Every target combination needs an anchor-admission control.')
        if any(not all(slot > 0 for slot in target) for target in targets):
            raise ValueError('Every target combination must be all-nonzero.')
        payload['_completion_selected'] = selected
        payload['_completion_combinations'] = tuple(combinations)
        payload['_completion_targets'] = tuple(targets)

    payload['_prompt_count'] = max(
        len(item['descriptions']) for item in classes)
    payload['_path'] = path
    return payload
