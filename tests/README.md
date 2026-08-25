# Public test suite

The tracked tests are the two Release 01 publication gates: the public Python
tree must not contain missing internal imports, and the portable Quick Start
training/evaluation configs must remain self-contained.

Run the lightweight public checks with:

```bash
uv run --no-sync pytest tests/test_internal_imports.py tests/test_portable_configs.py
```

Detailed development and benchmark regression tests remain in internal
worktrees and are excluded from the public branch.
