def get_vlm_model(config):
    """Select by explicit backbone type, independent of the checkpoint path."""
    from importlib import import_module

    kind = config.framework.qwenvl.type
    classes = {
        "qwen35": "starVLA.model.modules.vlm.QWen3_5:_QWen3_5_VL_Interface",
        "minicpm": "starVLA.model.modules.vlm.MiniCPMV:_MiniCPM_VL_Interface",
    }
    module, name = classes.get(kind, kind).split(":")
    return getattr(import_module(module), name)(config)
