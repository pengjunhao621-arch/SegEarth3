"""Contracts for the frozen role-functional SAM3 text screen."""

import json


PROTOCOL = 'role_functional_text_screen_v1'
SCHEMA_VERSION = 1
BANK_SCHEMA_VERSION = 1
PI_PROTOCOL = 'pi_role_compatibility_v1'
PI_SCHEMA_VERSION = 1

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


COMBINATION_VARIANTS = tuple(
    combo_variant_name(presence_slot, semantic_slot, instance_slot)
    for presence_slot in range(3)
    for semantic_slot in range(4)
    for instance_slot in range(3)
)

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

    payload['_prompt_count'] = max(
        len(item['descriptions']) for item in classes)
    payload['_path'] = path
    return payload
