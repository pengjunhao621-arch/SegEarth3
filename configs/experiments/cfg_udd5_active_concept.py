_base_ = '../cfg_udd5.py'

model = dict(
    seed_dataset_name='udd5',
    dump_active_concept_stats=False,
    use_active_concept_pruning=False,
)
