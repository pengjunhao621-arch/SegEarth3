_base_ = '../cfg_vaihingen.py'

model = dict(
    seed_dataset_name='vaihingen',
    dump_active_concept_stats=False,
    use_active_concept_pruning=False,
)
