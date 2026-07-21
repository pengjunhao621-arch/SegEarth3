_base_ = '../cfg_udd5.py'

model = dict(
    dump_state_action_atlas_stats=True,
    state_action_atlas_stats_path='logs/state_action_atlas_v1/udd5/atlas.jsonl',
    state_action_apply='baseline',
)
