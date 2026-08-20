# MiniCPM-V-4.6 Action Placeholder Utilities

## 推荐：placeholder + action head（不扩词表）

`<action> ◆ ◆ ◆ ◆ </action>` 里的 `◆` 是 **VLM hidden state 的 query 槽位**，不是离散动作 token：

- 拼进 user prompt（不是 assistant solution）
- forward 后在 `◆` 位置 gather hidden states → 送入 action model
- 动作 loss 在 action head 上，**不对 placeholder 做 LM 监督**

QwenOFT 在 Qwen3 上用 `🔍`（单 token）；MiniCPM-V-4.6 上 `🔍` 会拆成 3 段，因此默认改用词表里已有的单 token `◆`（id=158778）。`MiniCPMV.py` 启动时会校验 placeholder 必须为单 token。

### 配置

```yaml
framework:
  qwenvl:
    base_vlm: openbmb/MiniCPM-V-4.6
    action_placeholder_token: "◆"
```

### 代码示例

```python
vlm = get_vlm_model(config)
suffix = vlm.build_action_placeholder_suffix(num_placeholders=8)
instructions = [sample["lang"] + suffix for sample in batch]

inputs = vlm.build_qwenvl_inputs(images, instructions)
outputs = vlm(**inputs, output_hidden_states=True, return_dict=True)
queries = vlm.gather_action_placeholder_hidden_states(
    outputs.hidden_states[-1], inputs["input_ids"], num_placeholders=8
)
# queries: [B, 8, H] -> action_model
```

### 验证 placeholder 是否为单 token

```bash
python starVLA/model/modules/vlm/tools/add_minicpm_special_tokens/pick_unused_action_tokens.py \
  --model-id playground/Pretrained_models/MiniCPM-V-4.6 \
  --save-dir playground/Pretrained_models/MiniCPM-V-4.6 \
  --token "◆"
```

---

## 可选：离线扩词表

若实验性需要新增 special token，仍可使用 `add_special_tokens_to_minicpm.py`。

Requires `transformers>=5.7.0`.
