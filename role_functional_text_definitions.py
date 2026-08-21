"""Contracts for the frozen role-functional SAM3 text screen."""

import json
from itertools import product


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
ROLE_VISUAL_FIELD_PROTOCOL = 'role_visual_field_v1'
ROLE_VISUAL_FIELD_SCHEMA_VERSION = 1
ROLE_VISUAL_FIELD_COMPOSITIONS = tuple(
    ''.join(value) for value in product('FC', repeat=3))
ROLE_VISUAL_FIELD_VARIANT_NAMES = (
    'rvf_official',
    'rvf_best_text',
    *(f'rvf_anchor_{name}' for name in ROLE_VISUAL_FIELD_COMPOSITIONS),
    *(f'rvf_text_{name}' for name in ROLE_VISUAL_FIELD_COMPOSITIONS),
)

GLOBAL_LOCAL_EVIDENCE_PROTOCOL = 'global_local_evidence_v1'
GLOBAL_LOCAL_EVIDENCE_SCHEMA_VERSION = 1
GLOBAL_LOCAL_MIX_WEIGHTS = (
    ('g025', 0.25),
    ('g050', 0.50),
    ('g075', 0.75),
)
GLOBAL_LOCAL_EVIDENCE_VARIANT_NAMES = (
    'glv_official',
    'glv_best_text',
    'glv_anchor_local',
    'glv_anchor_global',
    'glv_text_local',
    'glv_text_global',
    *(f'glv_{family}_mix_{name}'
      for family in ('anchor', 'text')
      for name, _ in GLOBAL_LOCAL_MIX_WEIGHTS),
    'glv_anchor_max',
    'glv_text_max',
)

ROLE_MULTIMODAL_FUSION_PROTOCOL = 'role_multimodal_fusion_v1'
ROLE_MULTIMODAL_FUSION_SCHEMA_VERSION = 1
ROLE_MULTIMODAL_FUSION_BLEND = 0.5
ROLE_MULTIMODAL_FUSION_WINDOW = 3
ROLE_MULTIMODAL_FUSION_ROLE_SCOPES = ('p', 's', 'i', 'psi')
ROLE_MULTIMODAL_FUSION_ANCHOR_METHODS = (
    'v2_canvas_fine',
    'v2_canvas_shared',
    'v4_transport',
    'v5_agreement',
    'v6_qk',
    'v7_value',
    'v8_full',
)
ROLE_MULTIMODAL_FUSION_TEXT_METHODS = (
    'v1_role_text',
    *ROLE_MULTIMODAL_FUSION_ANCHOR_METHODS,
)
ROLE_MULTIMODAL_FUSION_VARIANT_NAMES = (
    'rmf_v0_official',
    *(f'rmf_{method}_anchor'
      for method in ROLE_MULTIMODAL_FUSION_ANCHOR_METHODS),
    *(f'rmf_{method}_{scope}'
      for method in ROLE_MULTIMODAL_FUSION_TEXT_METHODS
      for scope in ROLE_MULTIMODAL_FUSION_ROLE_SCOPES),
)

CONTEXT_RECOMPOSITION_PROTOCOL = 'context_recomposition_v1'
CONTEXT_RECOMPOSITION_SCHEMA_VERSION = 1
CONTEXT_RECOMPOSITION_BLEND = 0.5
CONTEXT_RECOMPOSITION_PROB_CLIP = 0.25
CONTEXT_RECOMPOSITION_LOGIT_CLIP = 1.0
CONTEXT_RECOMPOSITION_NEAR_RATIO = 0.25
CONTEXT_RECOMPOSITION_LOWPASS_DIVISOR = 16
CONTEXT_RECOMPOSITION_VARIANT_SPECS = (
    ('cr_official', 'protected', 'official'),
    ('cr_r0_native_role_text', 'r0', 'native_role_text'),
    ('cr_r1_final_residual', 'r1', 'final_context_residual'),
    ('cr_r2_semantic_residual', 'r2', 'semantic_context_residual'),
    ('cr_r3_instance_residual', 'r3', 'instance_context_residual'),
    ('cr_r4_si_native_presence', 'r4', 'si_native_presence'),
    ('cr_r5_lowfreq_semantic', 'r5', 'low_frequency_semantic'),
    ('cr_r6_foveated_si', 'r6', 'foveated_si'),
    ('cr_r7_foveated_ring', 'r7', 'foveated_ring'),
    ('cr_r8_role_context_did', 'r8', 'role_context_did'),
    ('cr_r9_surround_only', 'r9', 'surround_only'),
    ('cr_r9_target_scale_only', 'r9', 'target_scale_only'),
    ('cr_r10_near_only', 'r10', 'near_only'),
    ('cr_r10_far_only', 'r10', 'far_only'),
    ('cr_r11_prior_only', 'r11', 'class_prior_only'),
    ('cr_r11_spatial_only', 'r11', 'spatial_only'),
    ('cr_r12_add_only', 'r12', 'positive_correction_only'),
    ('cr_r12_suppress_only', 'r12', 'negative_correction_only'),
    ('cr_r13_geometry_only', 'r13', 'aligned_geometry_only'),
    ('cr_r13_scene_only', 'r13', 'geometry_permuted_scene_only'),
    ('cr_r14_logodds', 'r14', 'log_odds_context'),
    ('cr_r14_logodds_did', 'r14', 'log_odds_role_context_did'),
)
CONTEXT_RECOMPOSITION_VARIANT_NAMES = tuple(
    name for name, _, _ in CONTEXT_RECOMPOSITION_VARIANT_SPECS)

FUSION_AUDIT_PROTOCOL = 'overall_best_fusion_audit_v1'
FUSION_AUDIT_SCHEMA_VERSION = 1
FUSION_AUDIT_VARIANT_SPECS = (
    ('ofa_official', 'reference'),
    ('ofa_native', 'reference'),
    ('ofa_semantic_only', 'ablation'),
    ('ofa_instance_only', 'ablation'),
    ('ofa_takeover_i025', 'candidate'),
    ('ofa_takeover_i050', 'candidate'),
    ('ofa_takeover_i075', 'candidate'),
    ('ofa_smoothmax_t005', 'candidate'),
    ('ofa_smoothmax_t010', 'candidate'),
    ('ofa_soft_or', 'candidate'),
    ('ofa_instance_scale075', 'candidate'),
    ('ofa_instance_scale125', 'candidate'),
    ('ofa_branch_once', 'candidate'),
    ('ofa_no_outer_presence', 'candidate'),
    ('ofa_recompose_dense', 'candidate'),
    ('ofa_recompose_instance_supported', 'candidate'),
)
FUSION_AUDIT_VARIANT_NAMES = tuple(
    name for name, _ in FUSION_AUDIT_VARIANT_SPECS)
FUSION_AUDIT_MECHANISM_MAP_NAMES = (
    'anchor_semantic',
    'anchor_instance',
    'anchor_presence',
    'selected_semantic',
    'selected_instance',
    'selected_presence',
)

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
