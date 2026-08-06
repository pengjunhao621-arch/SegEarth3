"""Dependency-free contract for the SAM3 head-role prompt-conflict audit."""

SCHEMA_VERSION = 3
PROMPT_COUNT = 5

# Description zero is the canonical literal class name.  The four remaining
# descriptions come from the existing per-dataset prompt banks.  They are kept
# as fixed candidates rather than selected with ground truth.
DESCRIPTION_SLOTS = (
    (1, 'description_1'),
    (2, 'description_2'),
    (3, 'description_3'),
    (4, 'description_4'),
)

# Each tuple names the final-output path and the source used by
# (semantic, raw instance queries/object scores, Presence).  ``literal`` and
# ``description`` always mean complete, independently grounded SAM3 prompts;
# language tokens are never mixed between prompts.
HEAD_PATHS = (
    ('native', 'description', 'description', 'description'),
    ('semantic_only', 'description', 'literal', 'literal'),
    ('instance_only', 'literal', 'description', 'literal'),
    ('presence_only', 'literal', 'literal', 'description'),
    ('semantic_instance_literal_presence',
     'description', 'description', 'literal'),
)


def variant_name(description_slot, path_name):
    return f'd{int(description_slot)}_{path_name}'


VARIANT_NAMES = (
    'baseline',
    'literal_native',
    *(
        variant_name(description_slot, path_name)
        for description_slot, _ in DESCRIPTION_SLOTS
        for path_name, _, _, _ in HEAD_PATHS
    ),
)


CAUSAL_REFERENCE = 'literal_native'
