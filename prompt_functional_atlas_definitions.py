"""Dependency-free contract for the SAM3 prompt functional atlas."""

SCHEMA_VERSION = 2
PROMPT_COUNT = 5

PROMPT_SLOTS = (
    ('prompt_0_literal', 'literal'),
    ('prompt_1_short_domain', 'short_domain'),
    ('prompt_2_short_visual', 'short_visual'),
    ('prompt_3_context_phrase', 'context_phrase'),
    ('prompt_4_long_description', 'long_description'),
)

VARIANT_NAMES = (
    'baseline',
    *(name for name, _ in PROMPT_SLOTS),
    'pool_mean',
    'pool_max',
    'anchor_output',
    'visual_output',
    'entropy_output',
    'full_output',
    'full_bg_literal_output',
    'presence_selected_output',
    'presence_weighted_output',
    'e2e_output',
    'e2e_bg_literal_output',
    'literal_regrounded',
    'anchor_residual_010_regrounded',
    'anchor_residual_025_regrounded',
    'full_sequence_regrounded',
)

REGROUND_VARIANTS = (
    'literal_regrounded',
    'anchor_residual_010_regrounded',
    'anchor_residual_025_regrounded',
    'full_sequence_regrounded',
)
