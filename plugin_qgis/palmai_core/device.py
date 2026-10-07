# -*- coding: utf-8 -*-


def resolve_device(preference):
    """Map a UI preference ('gpu' | 'cpu' | explicit torch device) to a torch device string."""
    pref = (preference or "cpu").lower()
    if pref == "cpu":
        return "cpu"
    if pref not in ("gpu", "cuda", "mps"):
        return preference  # explicit device such as 'cuda:0'
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
