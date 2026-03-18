# Lazy imports to avoid pulling torch at package load time
from .features import augment_features

PRIMITIVE_TARGETS = ["CL_0", "CL_alpha", "CM_0", "CM_alpha", "CD0_wing", "CD0_body", "Cn_beta"]


def __getattr__(name):
    if name == "SurrogateModel":
        from .model import SurrogateModel
        return SurrogateModel
    if name == "reconstruct_aero":
        from .reconstruct import reconstruct_aero
        return reconstruct_aero
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
