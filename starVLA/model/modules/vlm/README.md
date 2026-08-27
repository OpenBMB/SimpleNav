# Release 01 VLM backend

The public Release 01 model set uses the Qwen3.5-VL navigation path in
`QWen3_5.py` for all released training and evaluation configs, including
EVT-Bench Track-DT.

`get_vlm_model(config)` selects the backend from
`config.framework.qwenvl.base_vlm` and rejects models outside this release.
