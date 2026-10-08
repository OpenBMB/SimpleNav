
# start policy server


```bash


your_ckpt=./results/Checkpoints/1003_qwenfast/checkpoints/steps_50000_pytorch_model.pt

uv run --no-sync python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port 10093 \
    --use_bf16
```


# connect to policy server for debug

```bash
uv run --no-sync python deployment/model_server/debug_server_policy.py

# plus server_policy.py into your vla controler by ref to debug_server_policy.py
```