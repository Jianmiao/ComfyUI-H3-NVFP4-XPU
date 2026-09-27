"""ComfyUI H3 NVFP4 XPU loader.

The node keeps the native MiniMax H3 encoder and quantization code, but stages
file-backed NVFP4 tensors through CPU memory before each XPU transfer. This is
needed on Windows XPU where the default whole-file mapping path can exhaust
commit memory or stall on packed tensor copies.
"""

import logging

import comfy.model_management
import comfy_aimdo.model_vbar


log = logging.getLogger("H3-NVFP4-XPU")


def _install_xpu_vbar_fallback():
    """Degrade to normal staged copies when XPU VBAR mapping is unavailable.

    The Intel Level Zero VBAR backend can initialize successfully but still
    fail its first virtual-memory map on some Windows drivers. ComfyUI already
    treats a ``None`` VBAR signature as the ordinary cast-buffer path, so make
    the backend failure recoverable instead of aborting VAE/weight execution.
    """
    if not comfy.model_management.is_intel_xpu():
        return

    model_vbar = comfy_aimdo.model_vbar
    if getattr(model_vbar, "_h3_xpu_vbar_fallback_installed", False):
        return

    native_vbar_fault = model_vbar.vbar_fault
    state = {"disabled": False, "warned": False}

    def vbar_fault_with_fallback(alloc):
        if state["disabled"]:
            return None
        try:
            return native_vbar_fault(alloc)
        except RuntimeError as error:
            if str(error) != "Fault failed: 2":
                raise
            state["disabled"] = True
            if not state["warned"]:
                log.warning(
                    "XPU VBAR mapping is unavailable; falling back to regular "
                    "staged weight transfers for this process."
                )
                state["warned"] = True
            return None

    model_vbar.vbar_fault = vbar_fault_with_fallback
    model_vbar._h3_xpu_vbar_fallback_installed = True


_install_xpu_vbar_fallback()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
