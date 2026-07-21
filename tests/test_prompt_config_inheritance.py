import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _top_level_assignments(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name) and target.id != "_base_"
    }


def test_prompt_experiment_sibling_bases_have_no_duplicate_keys():
    base_config = ROOT / "configs/base_config.py"
    dataset_config = ROOT / "configs/_base_/datasets/openearthmap_prompt.py"

    duplicates = _top_level_assignments(base_config) & _top_level_assignments(
        dataset_config
    )

    assert not duplicates, (
        "MMEngine forbids duplicate keys among sibling _base_ configs: "
        f"{sorted(duplicates)}"
    )
