# Release 01 VLM backends

The public Release 01 model set contains the two backends used by released
portable evaluation configs:

- `QWen3_5.py`: the primary Qwen3.5-VL navigation path;
- `MiniCPMV.py`: the MiniCPM-V path used by the UAV-Flow/EVT-Bench checkpoint.

`get_vlm_model(config)` selects the backend from
`config.framework.qwenvl.base_vlm` and rejects models outside this release.
