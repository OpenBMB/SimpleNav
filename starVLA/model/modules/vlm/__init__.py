def get_vlm_model(config):
    """Build one of the VLM backbones used by Release 01 checkpoints."""
    vlm_name = config.framework.qwenvl.base_vlm

    if "Qwen3.5" in vlm_name:
        from .QWen3_5 import _QWen3_5_VL_Interface

        return _QWen3_5_VL_Interface(config)
    if "minicpm" in vlm_name.lower():
        from .MiniCPMV import _MiniCPM_VL_Interface

        return _MiniCPM_VL_Interface(config)
    raise NotImplementedError(
        f"VLM model {vlm_name!r} is not part of the Release 01 public model set"
    )
