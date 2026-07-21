_base_ = '../cfg_loveda.py'

model = dict(
    dump_state_action_atlas_stats=True,
    state_action_atlas_stats_path='logs/state_action_atlas_v1/loveda/atlas.jsonl',
    state_action_apply='baseline',
)
