_base_ = '../cfg_iSAID.py'

model = dict(
    seed_dataset_name='isaid',
    dump_active_concept_stats=False,
    use_active_concept_pruning=False,
)
