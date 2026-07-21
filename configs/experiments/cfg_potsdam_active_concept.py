_base_ = '../cfg_potsdam.py'

model = dict(
    seed_dataset_name='potsdam',
    dump_active_concept_stats=False,
    use_active_concept_pruning=False,
)
