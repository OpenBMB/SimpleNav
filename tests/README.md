# Public test suite

The tracked tests cover public data adapters, LeRobot loading and validation, model/action contracts, portable training configs, checkpoint resume behavior, and benchmark evaluation adapters. They are regression checks for repository interfaces; they are not required to import or run SimpleNAV.

Run the lightweight public checks with:

```bash
uv run --no-sync pytest tests/test_internal_imports.py tests/test_portable_configs.py
```

Benchmark-specific and data-tool test selections are listed in [`NavVLAeval/README.md`](../NavVLAeval/README.md) and [`tool/navvla/README.md`](../tool/navvla/README.md). Tests for local-only experiment configs remain excluded by `.gitignore`.
