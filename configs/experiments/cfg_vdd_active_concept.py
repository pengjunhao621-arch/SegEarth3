_base_ = '../cfg_vdd.py'

model = dict(
    seed_dataset_name='vdd',
    dump_active_concept_stats=False,
    use_active_concept_pruning=False,
)
