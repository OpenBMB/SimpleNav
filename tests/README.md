# Public test suite

The tracked tests check repository imports and keep the portable Quick Start
training/evaluation configs self-contained.

Run the lightweight public checks with:

```bash
uv run --no-sync pytest tests/test_internal_imports.py tests/test_portable_configs.py
```

Additional tests live alongside their respective modules.
