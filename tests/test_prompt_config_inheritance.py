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


def _dict_assignment_keyword(path, assignment_name, keyword_name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == assignment_name
            for target in node.targets
        ):
            continue
        assert isinstance(node.value, ast.Call)
        for keyword in node.value.keywords:
            if keyword.arg == keyword_name:
                return keyword.value
    raise AssertionError(
        f"Could not find {assignment_name}.{keyword_name} in {path}"
    )


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


def test_prompt_experiment_defines_one_training_padding_policy():
    experiment = (
        ROOT / "configs/experiments/cfg_openearthmap_prompt_sam3_global.py"
    )
    preprocessor_node = _dict_assignment_keyword(
        experiment, "model", "data_preprocessor"
    )
    assert isinstance(preprocessor_node, ast.Call)
    preprocessor = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in preprocessor_node.keywords
    }

    has_size = preprocessor.get("size") is not None
    has_size_divisor = preprocessor.get("size_divisor") is not None
    assert has_size ^ has_size_divisor, (
        "MMSeg training requires exactly one of size and size_divisor"
    )
    assert preprocessor["size"] == (512, 512)
