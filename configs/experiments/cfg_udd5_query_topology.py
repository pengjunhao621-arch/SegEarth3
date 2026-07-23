_base_ = '../cfg_udd5.py'

model = dict(
    seed_dataset_name='udd5',
    dump_query_topology_stats=True,
    query_topology_stats_path='logs/query_topology/udd5/topology.jsonl',
    raw_mask_oracle_topk=24,
    query_topology_max_side=256,
    query_topology_max_regions=256,
    query_topology_min_pixels=16,
    query_topology_support_threshold=0.05,
    query_topology_mask_threshold=0.50,
    query_topology_semantic_threshold=0.35,
    query_topology_action_min_overlap=0.05,
    query_topology_action_padding=4,
    query_topology_include_background=True,
    query_topology_save_npz=True,
    query_topology_artifact_dir='logs/query_topology/udd5/artifacts',
    query_topology_max_saved_images=16,
)
