_base_ = '../cfg_loveda.py'

model = dict(
    seed_dataset_name='loveda',
    presence_allocation_dataset_name='loveda',
    dump_presence_allocation_stats=True,
    presence_allocation_stats_path=(
        'logs/presence_allocation/loveda/presence_allocation.jsonl'),
    presence_allocation_raw_gate_threshold=None,
    presence_allocation_support_threshold=0.05,
    presence_allocation_chunk_size=16,
    presence_allocation_logit_eps=1e-4,
    presence_allocation_strict_integrity=True,
    presence_allocation_integrity_tolerance=1e-5,
    presence_allocation_save_npz=True,
    presence_allocation_artifact_dir=(
        'logs/presence_allocation/loveda/artifacts'),
    presence_allocation_artifact_max_side=128,
    presence_allocation_max_saved_images=24,
)
