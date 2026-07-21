_base_ = '../cfg_iSAID.py'

model = dict(
    reviewer_dataset_name='isaid',
    reviewer_max_side=512,
    reviewer_cache_samples_per_image=4096,
    reviewer_cache_hard_keep_margin=0.15,
    reviewer_topk=3,
    reviewer_local_kernel=7,
    dump_reviewer_cache=False,
    use_learned_reviewer=False,
    dump_learned_reviewer_stats=False,
)
