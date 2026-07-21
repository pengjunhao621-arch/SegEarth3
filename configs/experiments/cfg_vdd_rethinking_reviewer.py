_base_ = '../cfg_vdd.py'

model = dict(
    reviewer_dataset_name='vdd',
    reviewer_max_side=512,
    reviewer_cache_samples_per_image=4096,
    reviewer_topk=3,
    reviewer_local_kernel=7,
    dump_reviewer_cache=False,
    use_learned_reviewer=False,
    dump_learned_reviewer_stats=False,
)
