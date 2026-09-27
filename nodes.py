import copy
import dataclasses
import json
import logging
import mmap
import os
import threading
from pathlib import Path

import torch

import comfy.hooks
import comfy.memory_management
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.sd
import comfy.utils
import folder_paths
from comfy.quant_ops import QuantizedTensor


log = logging.getLogger("H3-NVFP4-XPU")


class _MappedCheckpoint:
    def __init__(self, path):
        self.path = path
        self.file = open(path, "rb")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.lock = threading.Lock()
        self.header_size = int.from_bytes(self.mapping[:8], "little")
        self.header = json.loads(self.mapping[8 : 8 + self.header_size])
        self.data_offset = 8 + self.header_size

    def close(self):
        mapping, file = self.mapping, self.file
        self.mapping = None
        self.file = None
        if mapping is not None:
            mapping.close()
        if file is not None:
            file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _mapped_state(owner):
    state = {}
    for name, info in owner.header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = comfy.utils._TYPES[info["dtype"]]
        if start == end:
            state[name] = torch.empty(info["shape"], dtype=dtype)
            continue

        view = memoryview(owner.mapping)[owner.data_offset + start : owner.data_offset + end]
        tensor = torch.frombuffer(view, dtype=dtype).view(info["shape"])
        storage = tensor.untyped_storage()
        storage._comfy_tensor_file_slice = comfy.memory_management.TensorFileSlice(
            owner.file, owner.lock, owner.data_offset + start, end - start
        )
        storage._comfy_tensor_mmap_refs = (owner.mapping, view)
        state[name] = tensor
    return state


def _stage_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        return tensor

    staged = torch.empty_like(tensor, device="cpu")
    if not comfy.memory_management.read_tensor_file_slice_into(tensor, staged):
        staged.copy_(tensor)
    return staged


def _stage_quantized_weight(weight):
    if not isinstance(weight, QuantizedTensor) or weight.device.type != "cpu":
        return weight

    params = weight._params
    staged_params = {}
    for field in dataclasses.fields(params):
        value = getattr(params, field.name)
        if isinstance(value, torch.Tensor):
            staged_params[field.name] = _stage_tensor(value)
    params = dataclasses.replace(params, **staged_params)
    return QuantizedTensor(_stage_tensor(weight._qdata), weight._layout_cls, params)


def _staged_proxy(module, device):
    weight = getattr(module, "weight", None)
    if not isinstance(weight, QuantizedTensor) or weight.device.type != "cpu" or device.type != "xpu":
        return module

    proxy = copy.copy(module)
    proxy._parameters = module._parameters.copy()
    proxy._parameters["weight"] = torch.nn.Parameter(
        _stage_quantized_weight(weight), requires_grad=False
    )
    return proxy


def _make_staged_operations(quantization_metadata, compute_dtype):
    base = comfy.ops.mixed_precision_ops(
        quantization_metadata,
        compute_dtype,
        full_precision_mm=True,
    )

    class StagedOperations(base):
        class Linear(base.Linear):
            def forward_comfy_cast_weights(
                self,
                input,
                compute_dtype=None,
                want_requant=False,
                weight_only_quant=False,
            ):
                proxy = _staged_proxy(self, input.device)
                return base.Linear.forward_comfy_cast_weights(
                    proxy,
                    input,
                    compute_dtype,
                    want_requant=want_requant,
                    weight_only_quant=weight_only_quant,
                )

        class Embedding(base.Embedding):
            def forward_comfy_cast_weights(self, input, out_dtype=None):
                proxy = _staged_proxy(self, input.device)
                return base.Embedding.forward_comfy_cast_weights(
                    proxy,
                    input,
                    out_dtype=out_dtype,
                )

    return StagedOperations


class _H3StreamingPatcher(comfy.model_patcher.ModelPatcher):
    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        return super().load(
            device_to=device_to,
            lowvram_model_memory=0.1,
            force_patch_weights=force_patch_weights,
            full_load=False,
        )


def _load_clip(path):
    owner = _MappedCheckpoint(path)
    state = _mapped_state(owner)
    detected = comfy.sd.llama_detect([state])
    if detected.get("llama_quantization_metadata") is None:
        owner.close()
        raise ValueError("The selected text encoder has no ComfyUI quantization metadata")

    if not comfy.model_management.is_intel_xpu():
        owner.close()
        raise RuntimeError("H3 NVFP4 XPU loader requires an Intel XPU device")

    device = comfy.model_management.get_torch_device()
    operations = _make_staged_operations(
        detected["llama_quantization_metadata"],
        detected.get("dtype_llama", torch.bfloat16),
    )
    clip = comfy.sd.load_text_encoder_state_dicts(
        [state],
        clip_type=comfy.sd.CLIPType.MINIMAX,
        disable_dynamic=True,
        model_options={
            "custom_operations": operations,
            "load_device": device,
            "offload_device": torch.device("cpu"),
            "initial_device": torch.device("cpu"),
        },
    )

    patcher = _H3StreamingPatcher(
        clip.cond_stage_model,
        load_device=device,
        offload_device=torch.device("cpu"),
    )
    patcher.set_model_compute_dtype(torch.float32)
    patcher.is_clip = True
    patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
    patcher.set_attachments("h3_nvfp4_source", owner)
    clip.patcher = patcher
    log.info(
        "Loaded H3 NVFP4 XPU encoder with streamed CPU staging: %s -> %s",
        Path(path).name,
        device,
    )
    return clip


class H3NVFP4XPUClipLoader:
    @classmethod
    def INPUT_TYPES(cls):
        names = [
            name
            for name in folder_paths.get_filename_list("text_encoders")
            if "minimax_h3" in name.lower() and "nvfp4" in name.lower()
        ]
        return {"required": {"clip_name": (sorted(names),)}}

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/H3 XPU"

    def load_clip(self, clip_name):
        path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        return (_load_clip(path),)


NODE_CLASS_MAPPINGS = {
    "H3NVFP4XPUClipLoader": H3NVFP4XPUClipLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3NVFP4XPUClipLoader": "Load H3 NVFP4 (XPU streamed)",
}
