_base_ = '../cfg_vdd.py'

model = dict(
    seed_dataset_name='vdd',
    dump_concept_specificity_stats=False,
    use_concept_specificity_pruning=False,
    concept_specificity_max_side=512,
)
