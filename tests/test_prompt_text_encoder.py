import torch

from sam3.model.text_encoder_ve import TextTransformer


def test_external_embedding_path_matches_token_id_path():
    torch.manual_seed(11)
    encoder = TextTransformer(
        context_length=8,
        vocab_size=32,
        width=16,
        heads=4,
        layers=2,
        output_dim=16,
        output_tokens=True,
        pool_type="argmax",
        use_act_checkpoint=False,
    ).eval()
    token_ids = torch.tensor([[1, 4, 31, 0, 0, 0, 0, 0]])
    inputs_embeds = encoder.token_embedding(token_ids)
    with torch.no_grad():
        pooled_ids, tokens_ids = encoder(token_ids)
        pooled_embeds, tokens_embeds = encoder.forward_embeddings(
            token_ids, inputs_embeds
        )
    torch.testing.assert_close(pooled_ids, pooled_embeds)
    torch.testing.assert_close(tokens_ids, tokens_embeds)


def test_external_embedding_path_backpropagates_to_residual_only():
    torch.manual_seed(13)
    encoder = TextTransformer(
        context_length=6,
        vocab_size=16,
        width=8,
        heads=2,
        layers=1,
        output_dim=8,
        output_tokens=True,
        pool_type="argmax",
        use_act_checkpoint=False,
    ).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    token_ids = torch.tensor([[1, 3, 15, 0, 0, 0]])
    residual = torch.zeros(1, 6, 8, requires_grad=True)
    inputs_embeds = encoder.token_embedding(token_ids).detach() + residual
    pooled, tokens = encoder.forward_embeddings(token_ids, inputs_embeds)
    (pooled.square().mean() + tokens.square().mean()).backward()
    assert residual.grad is not None
    assert residual.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in encoder.parameters())

