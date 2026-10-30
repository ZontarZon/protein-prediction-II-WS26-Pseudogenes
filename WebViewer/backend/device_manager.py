"""Discover devices visible to PyTorch and resolve run requests safely."""

import torch


def available_devices():
    devices = [{"id": "cpu", "type": "cpu", "backend": "cpu", "name": "CPU", "label": "CPU"}]

    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                try:
                    name = torch.cuda.get_device_name(index)
                except (RuntimeError, AssertionError):
                    name = f"CUDA device {index}"
                devices.append({"id": f"cuda:{index}", "type": "gpu", "backend": "cuda", "index": index,
                                "name": name, "label": f"GPU {index} - {name}"})
    except (RuntimeError, AssertionError):
        pass

    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            for index in range(xpu.device_count()):
                try:
                    name = xpu.get_device_name(index)
                except (AttributeError, RuntimeError, AssertionError):
                    name = f"XPU device {index}"
                devices.append({"id": f"xpu:{index}", "type": "gpu", "backend": "xpu", "index": index,
                                "name": name, "label": f"GPU {index} - {name} (XPU)"})
    except (AttributeError, RuntimeError, AssertionError):
        pass

    try:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            devices.append({"id": "mps", "type": "gpu", "backend": "mps", "name": "Apple Metal (MPS)",
                            "label": "GPU - Apple Metal (MPS)"})
    except (AttributeError, RuntimeError):
        pass

    return devices


def default_device(devices=None):
    devices = devices or available_devices()
    return next((item for item in devices if item["type"] == "gpu"), devices[0])


def resolve_device(device_id):
    devices = available_devices()
    requested = device_id or default_device(devices)["id"]
    selected = next((item for item in devices if item["id"] == requested), None)
    if selected:
        return selected, None
    fallback = devices[0]
    return fallback, {"requested": requested, "selected": fallback["id"], "reason": "Requested device is no longer available; using CPU."}
