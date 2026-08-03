"""Dependency-free schema shared by runtime and offline summarization."""

SCHEMA_VERSION = 1

VARIANT_NAMES = (
    'baseline',
    'pool_max',
    'pool_mean',
    'anchor_output',
    'uniform_output',
    'visual_output',
    'entropy_output',
    'full_output',
    'full_no_anchor_output',
    'full_no_presence_gate_output',
    'anchor_regrounded',
    'uniform_regrounded',
    'visual_regrounded',
    'full_regrounded_surrogate',
    'full_regrounded_e2e',
)
