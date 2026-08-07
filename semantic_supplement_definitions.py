"""Dependency-free contracts for the two-prompt semantic-screen audit."""

import json


PROTOCOL = 'semantic_supplement_screen_v1'
SCHEMA_VERSION = 1
BANK_SCHEMA_VERSION = 1
MAX_CANDIDATES = 3

FAMILIES = (
    ('lexical', 'lexical_equivalents'),
    ('modifier', 'semantic_modifiers'),
)

CANDIDATE_PATHS = (
    'shared_native',
    'semantic_replace',
    'semantic_residual',
)

AGGREGATE_PATHS = (
    'mean_semantic_replace',
    'mean_semantic_residual',
)


def candidate_variant_name(family, slot, path):
    return f'{family}_s{int(slot)}_{path}'


def aggregate_variant_name(family, path):
    return f'{family}_{path}'


VARIANT_NAMES = (
    'baseline',
    'anchor_native',
    *(
        candidate_variant_name(family, slot, path)
        for family, _ in FAMILIES
        for slot in range(1, MAX_CANDIDATES + 1)
        for path in CANDIDATE_PATHS
    ),
    *(
        aggregate_variant_name(family, path)
        for family, _ in FAMILIES
        for path in AGGREGATE_PATHS
    ),
)

CAUSAL_REFERENCE = 'anchor_native'
FIXED_FAMILY_VARIANTS = tuple(
    aggregate_variant_name(family, 'mean_semantic_residual')
    for family, _ in FAMILIES
)


def _normalise_prompt(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a non-empty string.')
    return ' '.join(value.strip().split())


def load_semantic_supplement_bank(path, expected_class_names=None):
    """Load and validate a frozen, dataset-specific two-family prompt bank."""
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    if int(payload.get('schema_version', -1)) != BANK_SCHEMA_VERSION:
        raise ValueError(
            f'Prompt bank {path} has schema_version='
            f'{payload.get("schema_version")!r}; expected '
            f'{BANK_SCHEMA_VERSION}.')
    if payload.get('protocol') != PROTOCOL:
        raise ValueError(
            f'Prompt bank {path} must declare protocol={PROTOCOL!r}.')
    if payload.get('frozen_before_evaluation') is not True:
        raise ValueError(
            f'Prompt bank {path} must set frozen_before_evaluation=true.')
    for policy in ('lexical_policy', 'modifier_policy'):
        _normalise_prompt(payload.get(policy), policy)
    classes = payload.get('classes')
    if not isinstance(classes, list) or not classes:
        raise ValueError(f'Prompt bank {path} needs a non-empty classes list.')
    ids = [int(item.get('id', -1)) for item in classes]
    if ids != list(range(len(classes))):
        raise ValueError(
            f'Prompt bank {path} class IDs must be contiguous and ordered.')

    observed_names = []
    for item in classes:
        name = _normalise_prompt(item.get('name'), 'class name')
        anchor = _normalise_prompt(item.get('anchor'), f'{name}.anchor')
        if anchor.lower() != name.lower():
            raise ValueError(
                f'{name}: anchor must equal the canonical class name.')
        observed_names.append(name.lower())
        all_prompts = [anchor]
        for _, field in FAMILIES:
            candidates = item.get(field, [])
            if not isinstance(candidates, list):
                raise ValueError(f'{name}.{field} must be a list.')
            if len(candidates) > MAX_CANDIDATES:
                raise ValueError(
                    f'{name}.{field} has {len(candidates)} candidates; '
                    f'maximum is {MAX_CANDIDATES}.')
            normalised = [
                _normalise_prompt(value, f'{name}.{field}')
                for value in candidates
            ]
            if len({value.lower() for value in normalised}) != len(normalised):
                raise ValueError(f'{name}.{field} contains duplicates.')
            if any(value.lower() == anchor.lower() for value in normalised):
                raise ValueError(
                    f'{name}.{field} must not repeat the anchor.')
            if field == 'semantic_modifiers' and any(
                    anchor.lower() not in value.lower()
                    for value in normalised):
                raise ValueError(
                    f'Every {name}.{field} candidate must retain the anchor.')
            item[field] = normalised
            all_prompts.extend(normalised)
        if len({value.lower() for value in all_prompts}) != len(all_prompts):
            raise ValueError(
                f'{name}: candidate families must not overlap.')
        item['name'] = name
        item['anchor'] = anchor
        # Compatibility with the shared SAM3 text-cache helper.  Candidate
        # counts remain variable; missing slots are exact anchor fallbacks.
        item['descriptions'] = all_prompts

    if expected_class_names is not None:
        expected = [str(value).strip().lower()
                    for value in expected_class_names]
        if observed_names != expected:
            raise ValueError(
                f'Prompt bank class order mismatch. expected={expected}, '
                f'observed={observed_names}')
    payload['_prompt_count'] = 1 + len(FAMILIES) * MAX_CANDIDATES
    payload['_path'] = path
    return payload
