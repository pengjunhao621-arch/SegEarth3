# RemoteCLIP checkpoint

Download the official `RemoteCLIP-ViT-L-14.pt` checkpoint from:

- https://huggingface.co/chendelong/RemoteCLIP/blob/main/RemoteCLIP-ViT-L-14.pt

Place the binary checkpoint at:

```text
weights/remoteclip/RemoteCLIP-ViT-L-14.pt
```

The checkpoint file is intentionally not tracked or synchronized. The planned
prompt-training wrapper will receive this path from configuration rather than
using SCORE's working-directory-dependent `RemoteCLIP/` hard-coded path.
