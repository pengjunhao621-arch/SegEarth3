_base_ = '../cfg_loveda.py'

model = dict(
    seed_dataset_name='loveda',
    dump_active_concept_stats=False,
    use_active_concept_pruning=False,
)
