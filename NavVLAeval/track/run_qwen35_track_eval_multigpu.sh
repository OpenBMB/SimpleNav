#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
if (( $# < 1 )); then
  echo "usage: $0 {at|dt|stt} [extra eval args]" >&2
  exit 2
fi
task="$1"
shift
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Keep each Habitat process short-lived.  1405 AT episodes / 175 splits is
# roughly eight episodes per simulator instance, which avoids long-lived EGL
# renderers becoming unstable.
split_num="${SPLIT_NUM:-281}"
split_id="${SPLIT_ID:-all}"
save_path="${SAVE_PATH:-${repo_root}/local/eval_results/evt_bench/track_qwen35_${task}}"

IFS=',' read -r -a gpus <<< "${gpu_list}"
if (( ${#gpus[@]} == 0 )); then
  echo "CUDA_VISIBLE_DEVICES is empty" >&2
  exit 2
fi

mkdir -p "${save_path}/logs"

split_ids=()
if [[ "${split_id}" == "all" || "${split_id}" == "ALL" || "${ALL_SPLITS:-0}" == "1" ]]; then
  for ((current_split_id = 0; current_split_id < split_num; current_split_id++)); do
    split_ids+=("${current_split_id}")
  done
else
  IFS=',' read -r -a split_ids <<< "${split_id}"
fi

worker() {
  local worker_id=$1
  local gpu_id=$2
  shift 2
  local current_split_id log_file

  for ((idx = worker_id; idx < ${#split_ids[@]}; idx += ${#gpus[@]})); do
    current_split_id="${split_ids[$idx]}"
    log_file="${save_path}/logs/split_${current_split_id}_gpu_${gpu_id}.log"
    echo "[eval] task=${task} split=${current_split_id}/${split_num} gpu=${gpu_id} log=${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" EGL_VISIBLE_DEVICES="${gpu_id}" EGL_DEVICE_ID=0 \
      "${script_dir}/run_qwen35_track_eval.sh" "${task}" \
      --save-path "${save_path}" --split-id "${current_split_id}" --split-num "${split_num}" \
      "$@" \
      >"${log_file}" 2>&1
    echo "[eval] task=${task} split=${current_split_id}/${split_num} gpu=${gpu_id} finished"
  done
}

echo "[eval] task=${task} splits=${#split_ids[@]}/${split_num} gpus=${gpus[*]} save=${save_path}"
pids=()
for worker_id in "${!gpus[@]}"; do
  worker "${worker_id}" "${gpus[$worker_id]}" "$@" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
