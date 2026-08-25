#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${PYTHON_BIN:-}
if [[ -z "${python_bin}" ]]; then
  python_bin=${repo_root}/.venv/bin/python
  if [[ ! -x "${python_bin}" ]]; then
    git_common_dir=$(git -C "${repo_root}" rev-parse --path-format=absolute --git-common-dir)
    python_bin=$(dirname "${git_common_dir}")/.venv/bin/python
  fi
fi
encoder_ckpt=${ENCODER_CKPT:-${repo_root}/local/models/Qwen3.5-4B}
profile=qwen3_5_4b_postmerge_pool4_256_mmap
batch_size=${BATCH_SIZE:-8}
prefetch_batches=${PREFETCH_BATCHES:-2}
shard_size=${SHARD_SIZE:-256}
num_gpus=${NUM_GPUS:-1}
limit=${LIMIT:-}
preflight_only=${PREFLIGHT_ONLY:-0}
min_free_gib=${MIN_FREE_GIB:-256}
selected_csv=${DATASETS:-AerialVLN,OpenFly,TravelUAV}
log_dir=${LOG_DIR:-${repo_root}/logs/qwen35_postmerge_pool4_cache}

datasets=(
  "AerialVLN|${repo_root}/local/data/AerialVLN_lerobot/vln_train|front"
  "OpenFly|${repo_root}/local/data/OpenFly_lerobot/vln_train|front"
  "TravelUAV|${repo_root}/local/data/TravelUAV_lerobot/vln_train|front,left,right,rear"
)

if [[ ! -x "${python_bin}" ]]; then
  echo "python is not executable: ${python_bin}" >&2
  exit 2
fi
if [[ ! -f "${encoder_ckpt}/config.json" || ! -f "${encoder_ckpt}/model.safetensors.index.json" ]]; then
  echo "incomplete Qwen3.5 checkpoint: ${encoder_ckpt}" >&2
  exit 2
fi
if ! [[ "${num_gpus}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: ${num_gpus}" >&2
  exit 2
fi
if [[ -n "${limit}" ]] && ! [[ "${limit}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be a positive integer, got: ${limit}" >&2
  exit 2
fi
if ! [[ "${shard_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARD_SIZE must be a positive integer, got: ${shard_size}" >&2
  exit 2
fi
if ! [[ "${min_free_gib}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MIN_FREE_GIB must be a positive integer, got: ${min_free_gib}" >&2
  exit 2
fi

IFS=',' read -r -a selected_names <<<"${selected_csv}"
selected_entries=()
for selected_name in "${selected_names[@]}"; do
  matched=0
  for entry in "${datasets[@]}"; do
    IFS='|' read -r name _root _camera_csv <<<"${entry}"
    if [[ "${name}" == "${selected_name}" ]]; then
      selected_entries+=("${entry}")
      matched=1
      break
    fi
  done
  if (( ! matched )); then
    echo "unknown dataset in DATASETS: ${selected_name}" >&2
    exit 2
  fi
done

preflight_args=("${profile}" "${encoder_ckpt}" "${limit}" "${shard_size}" "${min_free_gib}")
for entry in "${selected_entries[@]}"; do
  IFS='|' read -r name root camera_csv <<<"${entry}"
  preflight_args+=("${name}" "${root}" "${camera_csv}")
done

PYTHONPATH="${repo_root}" "${python_bin}" - "${preflight_args[@]}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

profile, encoder_ckpt, limit_text, shard_size_text, min_free_gib_text, *values = sys.argv[1:]
limit = int(limit_text) if limit_text else None
shard_size = int(shard_size_text)
min_free_gib = int(min_free_gib_text)
checkpoint_root = Path(encoder_ckpt)
checkpoint_config = json.loads((checkpoint_root / "config.json").read_text(encoding="utf-8"))
vision_config = checkpoint_config.get("vision_config", {})
text_config = checkpoint_config.get("text_config", {})
checkpoint_contract = {
    "model_type": (checkpoint_config.get("model_type"), "qwen3_5"),
    "vision.hidden_size": (vision_config.get("hidden_size"), 1024),
    "vision.out_hidden_size": (vision_config.get("out_hidden_size"), 2560),
    "vision.patch_size": (vision_config.get("patch_size"), 16),
    "vision.spatial_merge_size": (vision_config.get("spatial_merge_size"), 2),
    "text.hidden_size": (text_config.get("hidden_size"), 2560),
}
checkpoint_mismatches = {
    key: values for key, values in checkpoint_contract.items() if values[0] != values[1]
}
if checkpoint_mismatches:
    raise SystemExit(f"incompatible Qwen3.5-4B checkpoint config: {checkpoint_mismatches}")
weight_index = json.loads((checkpoint_root / "model.safetensors.index.json").read_text(encoding="utf-8"))
missing_weight_shards = sorted(
    shard for shard in set(weight_index.get("weight_map", {}).values()) if not (checkpoint_root / shard).is_file()
)
if missing_weight_shards:
    raise SystemExit(f"Qwen3.5 checkpoint is missing weight shards: {missing_weight_shards}")
expected_manifest = {
    "name": profile,
    "visual_head": "qwen3_5_postmerge_pool4",
    "encoder_name": "Qwen3.5-4B",
    "encoder_ckpt": encoder_ckpt,
    "token_level": "vit_postmerge_pool4",
    "token_count": 4,
    "dtype": "uint16",
    "storage_encoding": "bfloat16_bits",
    "file_format": "mmap_npy",
    "cache_stage": "vit_postmerge_pool4",
    "input_resize": [256, 256],
    "patch_size": 16,
    "spatial_merge_size": 2,
    "shard_size": shard_size,
}
bytes_per_ref = 4 * 2560 * 2
total_required = 0
dataset_roots = []
for offset in range(0, len(values), 3):
    name, root_text, camera_csv = values[offset : offset + 3]
    root = Path(root_text)
    dataset_roots.append(root)
    required = [
        root / "meta/info.json",
        root / "meta/navvla_cameras.json",
        root / "meta/navvla_video_index.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"{name}: missing required dataset files: {missing}")
    cameras = json.loads((root / "meta/navvla_cameras.json").read_text(encoding="utf-8"))
    selected_cameras = camera_csv.split(",")
    unknown = sorted(set(selected_cameras) - set(cameras))
    if unknown:
        raise SystemExit(f"{name}: unknown cameras: {unknown}")
    video_keys = {str(cameras[camera]["video_key"]) for camera in selected_cameras}
    video_index = pq.read_table(root / "meta/navvla_video_index.parquet", columns=["video_key", "available"])
    camera_mask = pc.is_in(video_index["video_key"], value_set=pa.array(sorted(video_keys)))
    available_mask = pc.fill_null(video_index["available"], False)
    available = video_index.filter(pc.and_(camera_mask, available_mask))
    total_refs = len(available)

    profile_root = root / "cache/visual_tokens" / profile
    manifest_path = profile_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in expected_manifest.items()
            if manifest.get(key) != expected
        }
        if mismatches:
            raise SystemExit(f"{name}: existing target profile has incompatible manifest: {mismatches}")
    existing_refs = 0
    index_path = profile_root / "index.parquet"
    if index_path.exists():
        index_schema = pq.read_schema(index_path)
        if "camera_name" in index_schema.names:
            index_cameras = pq.read_table(index_path, columns=["camera_name"])
            existing_refs = len(
                index_cameras.filter(
                    pc.is_in(index_cameras["camera_name"], value_set=pa.array(selected_cameras))
                )
            )
    missing_refs = max(0, total_refs - existing_refs)
    planned_refs = min(missing_refs, limit) if limit is not None else missing_refs
    required_bytes = planned_refs * bytes_per_ref
    total_required += required_bytes
    print(
        f"preflight dataset={name} cameras={camera_csv} total_refs={total_refs} "
        f"existing_refs={existing_refs} planned_refs={planned_refs} raw_cache_tib={required_bytes / 2**40:.3f}",
        flush=True,
    )

usage = shutil.disk_usage(dataset_roots[0])
required_with_margin = int(total_required * 1.05) + min_free_gib * 2**30
print(
    f"preflight total_raw_tib={total_required / 2**40:.3f} "
    f"required_with_margin_tib={required_with_margin / 2**40:.3f} free_tib={usage.free / 2**40:.3f} "
    f"reserved_gib={min_free_gib}",
    flush=True,
)
if required_with_margin > usage.free:
    raise SystemExit(
        "insufficient free space for selected full cache plan; select datasets separately, set LIMIT for a smoke run, "
        "or provision more storage"
    )
PY

if [[ "${preflight_only}" == "1" ]]; then
  echo "Qwen3.5 post-merge pool4 cache preflight passed; no cache files were written"
  exit 0
fi

mkdir -p "${log_dir}"
for entry in "${selected_entries[@]}"; do
  IFS='|' read -r name root camera_csv <<<"${entry}"
  IFS=',' read -r -a camera_names <<<"${camera_csv}"
  visual_args=(
    "${root}"
    --profile "${profile}"
    --visual-head qwen3_5_postmerge_pool4
    --encoder-name Qwen3.5-4B
    --encoder-ckpt "${encoder_ckpt}"
    --token-level vit_postmerge_pool4
    --token-count 4
    --hidden-dim 0
    --dtype uint16
    --file-format mmap_npy
    --shard-size "${shard_size}"
    --input-resize 256x256
    --camera-names "${camera_names[@]}"
    --skip-existing
    --all-token-budgets
    --batch-size "${batch_size}"
    --prefetch-batches "${prefetch_batches}"
  )
  if [[ -n "${limit}" ]]; then
    visual_args+=(--limit "${limit}")
  else
    visual_args+=(--validate-after)
  fi

  command=("${python_bin}")
  if (( num_gpus > 1 )); then
    command+=(
      -m torch.distributed.run
      --standalone
      --nproc-per-node "${num_gpus}"
      --module
    )
  else
    command+=(-m)
  fi
  command+=(tool.navvla.cli.generate_visual_cache)

  log_path="${log_dir}/${name}.log"
  echo "[$(date --iso-8601=seconds)] dataset=${name} cameras=${camera_csv} profile=${profile}" | tee "${log_path}"
  PYTHONPATH="${repo_root}" "${command[@]}" "${visual_args[@]}" 2>&1 | tee -a "${log_path}"
done

echo "selected Qwen3.5 post-merge pool4 cache jobs completed; existing cache profiles were preserved"
