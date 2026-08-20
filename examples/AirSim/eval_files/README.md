# AirSim Eval Files Moved

The active NavVLA evaluation code now lives at the repository root:

```text
NavVLAeval/
```

Use commands such as:

```bash
bash NavVLAeval/openfly/run_eval.sh --dry-run
bash NavVLAeval/traveluav/run_eval.sh --dry-run
```

This legacy directory is kept only for historical evaluation outputs such as
`openfly/eval_result/` and local cache artifacts. Do not add new eval code here.
