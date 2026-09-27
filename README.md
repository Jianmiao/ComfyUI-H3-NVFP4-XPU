# ComfyUI H3 NVFP4 XPU

This custom node provides a streamed loader for the MiniMax H3 NVFP4/AWQ text
encoder on Intel XPU.

It keeps ComfyUI's native MiniMax H3 tokenizer, Qwen3-VL vision path,
quantization metadata, FP32 conditioning output, and model patcher contract. The
XPU-specific part is limited to file-backed tensor staging: packed weights and
their scales are read into ordinary CPU tensors one layer at a time before the
native ComfyUI operation moves them to XPU. No NVIDIA NVFP4 kernel is forced.

The node is:

```text
Load H3 NVFP4 (XPU streamed)
```

It exposes the NVFP4 H3 text encoder from the `text_encoders` model folder and
outputs a normal `CLIP` connection. Use this node in place of `CLIPLoader` for
the H3 Ref2VA workflow. The diffusion model and both VAEs remain the native H3
nodes.

Requirements:

- Intel XPU with a working `torch.xpu` ComfyUI installation.
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` in a configured
  `text_encoders` folder.
- ComfyUI's installed `comfy-kitchen` quantized tensor support.

The loader is intentionally H3/NVFP4-specific. It does not replace generic
NVFP4 support, claim native FP4 hardware execution, or alter
`supports_nvfp4_compute`. The first text and image conditioning passes can take
several minutes on an XPU because the 15.69GB checkpoint is streamed through
CPU staging. Full nine-image video generation has not been validated yet.

On Windows XPU drivers where the experimental DynamicVRAM VBAR map returns a
Level Zero error, this node automatically falls back to ComfyUI's regular
staged weight transfers. The H3 launcher uses `--xpu-vae-no-vbar`, which keeps
the video VAE on the GPU while using the stable legacy VAE patcher; DynamicVRAM
remains available for the diffusion model and text encoder.
