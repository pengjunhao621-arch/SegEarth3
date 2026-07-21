import torch

from prompt_synthesis import PromptSynthesisModule


def _inputs():
    torch.manual_seed(3)
    module = PromptSynthesisModule(
        spatial_dim=8,
        language_dim=8,
        text_width=16,
        global_dim=6,
        context_length=10,
        num_shared_tokens=4,
        num_state_prototypes=4,
        num_global_tokens=1,
    )
    spatial = torch.randn(1, 8, 5, 5)
    static_inputs = torch.randn(3, 10, 16)
    static_language = torch.randn(3, 10, 8)
    padding_mask = torch.zeros(3, 10, dtype=torch.bool)
    padding_mask[:, -2:] = True
    global_feature = torch.randn(1, 6)
    return module, spatial, static_inputs, static_language, padding_mask, global_feature


def test_static_mode_is_exact_fallback():
    module, spatial, static_inputs, static_language, mask, global_feature = _inputs()
    dynamic, diagnostics = module(
        spatial,
        static_inputs,
        static_language,
        mask,
        global_feature,
        component_mode="static_baseline",
    )
    torch.testing.assert_close(dynamic[0], static_inputs)
    assert diagnostics["gate_shared"].item() == 0.0
    assert diagnostics["gate_state"].item() == 0.0
    assert diagnostics["gate_global"].item() == 0.0


def test_component_interventions_only_disable_requested_gate():
    module, spatial, static_inputs, static_language, mask, global_feature = _inputs()
    expected_zero = {
        "no_shared": "gate_shared",
        "no_state": "gate_state",
        "no_global": "gate_global",
    }
    for mode, zero_key in expected_zero.items():
        _, diagnostics = module(
            spatial, static_inputs, static_language, mask, global_feature, mode
        )
        assert diagnostics[zero_key].item() == 0.0
        other = {
            "gate_shared",
            "gate_state",
            "gate_global",
        } - {zero_key}
        assert all(diagnostics[key].item() > 0.0 for key in other)


def test_all_prompt_components_receive_gradient():
    module, spatial, static_inputs, static_language, mask, global_feature = _inputs()
    dynamic, _ = module(
        spatial, static_inputs, static_language, mask, global_feature, "full"
    )
    dynamic.square().mean().backward()
    assert module.shared_context.grad is not None
    assert module.shared_context.grad.abs().sum() > 0
    assert module.state_slots.grad is not None
    assert module.state_slots.grad.abs().sum() > 0
    assert module.global_to_text[1].weight.grad is not None
    assert module.global_to_text[1].weight.grad.abs().sum() > 0
    assert module.gate_logits.grad is not None
    assert module.gate_logits.grad.abs().sum() > 0


def test_diagnostic_shapes_are_stable():
    module, spatial, static_inputs, static_language, mask, global_feature = _inputs()
    dynamic, diagnostics = module(
        spatial, static_inputs, static_language, mask, global_feature, "full"
    )
    assert dynamic.shape == (1, 3, 10, 16)
    assert diagnostics["attention"].shape == (1, 3, 4, 25)
    assert diagnostics["state_prototypes"].shape == (1, 3, 4, 8)
    assert 1.0 <= diagnostics["prototype_effective_rank"].item() <= 4.0

