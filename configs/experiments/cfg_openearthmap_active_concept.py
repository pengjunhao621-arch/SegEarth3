_base_ = '../cfg_openearthmap.py'

model = dict(
    seed_dataset_name='openearthmap',
    dump_active_concept_stats=False,
    use_active_concept_pruning=False,
)
