#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: bash $0 CONFIG_YAML [--dry-run|--preflight-only]" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

config_arg=$1
shift
dry_run=false
preflight_only=false
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 1 ]]; then
    usage
    exit 2
  fi
  case $1 in
    --dry-run) dry_run=true ;;
    --preflight-only) preflight_only=true ;;
    *) usage; exit 2 ;;
  esac
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=$(cd -- "${script_dir}/../../../.." && pwd)

if [[ ${config_arg} = /* ]]; then
  config_yaml=${config_arg}
else
  config_yaml=$(realpath -m -- "${config_arg}")
fi
if [[ ! -f ${config_yaml} ]]; then
  echo "ERROR: config file not found: ${config_yaml}" >&2
  exit 1
fi

bootstrap_python=${code_root}/.venv/bin/python
if [[ ! -x ${bootstrap_python} ]]; then
  echo "ERROR: repository Python not found: ${bootstrap_python}" >&2
  exit 1
fi

runtime_dir=$(mktemp -d "${TMPDIR:-/tmp}/navvla-train.XXXXXX")
cleanup() {
  rm -rf -- "${runtime_dir}"
}
trap cleanup EXIT

"${bootstrap_python}" - "${config_yaml}" "${runtime_dir}" "${code_root}" <<'PY'
from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

import yaml


config_path = Path(sys.argv[1])
runtime_dir = Path(sys.argv[2])
code_root = Path(sys.argv[3])


def require_mapping(parent: dict, key: str) -> dict:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"`{key}` must be a mapping in {config_path}")
    return value


def require_positive_int(parent: dict, key: str, path: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{path}.{key}` must be a positive integer")
    return value


def resolve_repo_path(value: str, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{key}` must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else code_root / path


with config_path.open(encoding="utf-8") as stream:
    source_config = yaml.safe_load(stream)
if not isinstance(source_config, dict):
    raise ValueError(f"training config must be a mapping: {config_path}")

launcher = require_mapping(source_config, "launcher")
deepspeed = copy.deepcopy(require_mapping(source_config, "deepspeed"))
datasets = require_mapping(source_config, "datasets")
vla_data = require_mapping(datasets, "vla_data")
trainer = require_mapping(source_config, "trainer")

mode = launcher.get("mode", "local")
if mode not in {"local", "slurm", "ssh"}:
    raise ValueError("`launcher.mode` must be `local`, `slurm`, or `ssh`")

num_machines = require_positive_int(launcher, "num_machines", "launcher")
num_processes = require_positive_int(launcher, "num_processes", "launcher")
main_process_port = require_positive_int(launcher, "main_process_port", "launcher")
if num_processes % num_machines != 0:
    raise ValueError("`launcher.num_processes` must be divisible by `launcher.num_machines`")
if mode == "local" and num_machines != 1:
    raise ValueError("`launcher.mode: local` requires `launcher.num_machines: 1`")
if mode == "slurm" and num_machines == 1:
    raise ValueError("`launcher.mode: slurm` requires more than one machine")
if mode == "ssh" and num_machines != 2:
    raise ValueError("`launcher.mode: ssh` currently requires exactly two machines")

per_device_batch_size = require_positive_int(vla_data, "per_device_batch_size", "datasets.vla_data")
gradient_accumulation_steps = require_positive_int(trainer, "gradient_accumulation_steps", "trainer")
global_batch_size = per_device_batch_size * gradient_accumulation_steps * num_processes

derived_deepspeed_keys = {
    "train_micro_batch_size_per_gpu",
    "gradient_accumulation_steps",
    "train_batch_size",
}
duplicates = sorted(derived_deepspeed_keys.intersection(deepspeed))
if duplicates:
    raise ValueError(
        "DeepSpeed batch settings are derived from the training config; remove duplicate keys: "
        + ", ".join(duplicates)
    )
deepspeed["train_micro_batch_size_per_gpu"] = per_device_batch_size
deepspeed["gradient_accumulation_steps"] = gradient_accumulation_steps
deepspeed["train_batch_size"] = global_batch_size

base_run_id = source_config.get("run_id")
if not isinstance(base_run_id, str) or not base_run_id.strip():
    raise ValueError("`run_id` must be a non-empty string")
run_id_override = os.environ.get("NAVVLA_RUN_ID_OVERRIDE")
if run_id_override:
    if re.fullmatch(r"[A-Za-z0-9._-]+", run_id_override) is None:
        raise ValueError("`NAVVLA_RUN_ID_OVERRIDE` contains unsupported characters")
    run_id = run_id_override
else:
    suffix_mode = launcher.get("run_id_suffix", "timestamp")
    if suffix_mode == "timestamp":
        run_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    elif suffix_mode == "slurm_job_id":
        run_suffix = os.environ.get("SLURM_JOB_ID")
        if not run_suffix:
            raise ValueError("`SLURM_JOB_ID` is required for `launcher.run_id_suffix: slurm_job_id`")
    elif suffix_mode in {None, "none"}:
        run_suffix = None
    else:
        raise ValueError("`launcher.run_id_suffix` must be `timestamp`, `slurm_job_id`, or `none`")
    run_id = f"{base_run_id}_{run_suffix}" if run_suffix else base_run_id

run_root_dir = resolve_repo_path(source_config.get("run_root_dir"), "run_root_dir")
python_bin = resolve_repo_path(launcher.get("python", ".venv/bin/python"), "launcher.python")
training_script = resolve_repo_path(
    launcher.get("training_script", "starVLA/training/train_starvla.py"),
    "launcher.training_script",
)

cuda_visible_devices = launcher.get("cuda_visible_devices", "all")
if not isinstance(cuda_visible_devices, str) or not cuda_visible_devices.strip():
    raise ValueError("`launcher.cuda_visible_devices` must be a non-empty string")
processes_per_machine = num_processes // num_machines
if cuda_visible_devices != "all":
    visible_devices = [device.strip() for device in cuda_visible_devices.split(",") if device.strip()]
    if len(visible_devices) < processes_per_machine:
        raise ValueError(
            "`launcher.cuda_visible_devices` exposes fewer devices than processes per machine "
            f"({len(visible_devices)} < {processes_per_machine})"
        )

master_addr = launcher.get("master_addr", "auto")
if not isinstance(master_addr, str) or not master_addr.strip():
    raise ValueError("`launcher.master_addr` must be a non-empty string")

environment = launcher.get("environment", {})
if not isinstance(environment, dict):
    raise ValueError("`launcher.environment` must be a mapping")
for key in environment:
    if not isinstance(key, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
        raise ValueError(f"invalid environment variable name: {key!r}")

ssh_hosts = []
rdma_preflight = None
if mode == "ssh":
    if master_addr == "auto":
        raise ValueError("`launcher.mode: ssh` requires an explicit InfiniBand `master_addr`")
    try:
        ipaddress.IPv4Address(master_addr)
    except ipaddress.AddressValueError as exc:
        raise ValueError("`launcher.master_addr` must be an IPv4 address for SSH mode") from exc

    forbidden_global_keys = {"NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_IB_HCA"}
    duplicates = sorted(forbidden_global_keys.intersection(environment))
    if duplicates:
        raise ValueError(
            "SSH mode requires per-host network bindings; remove global keys: " + ", ".join(duplicates)
        )
    if str(environment.get("NCCL_IB_DISABLE")) != "0":
        raise ValueError("SSH mode requires `launcher.environment.NCCL_IB_DISABLE: \"0\"`")
    if environment.get("NCCL_NET") != "IB":
        raise ValueError("SSH mode requires `launcher.environment.NCCL_NET: IB` to forbid socket fallback")
    if environment.get("NCCL_DEBUG") != "INFO":
        raise ValueError("SSH mode requires `launcher.environment.NCCL_DEBUG: INFO` for NET/IB evidence")
    debug_subsys = str(environment.get("NCCL_DEBUG_SUBSYS", ""))
    if "NET" not in {part.strip() for part in debug_subsys.split(",")}:
        raise ValueError("SSH mode requires NET in `launcher.environment.NCCL_DEBUG_SUBSYS`")

    ssh = require_mapping(launcher, "ssh")
    ssh_hosts = ssh.get("hosts")
    if not isinstance(ssh_hosts, list) or len(ssh_hosts) != num_machines:
        raise ValueError("`launcher.ssh.hosts` must contain exactly one entry per machine")
    normalized_hosts = []
    seen_aliases = set()
    for rank, host in enumerate(ssh_hosts):
        if not isinstance(host, dict):
            raise ValueError(f"`launcher.ssh.hosts[{rank}]` must be a mapping")
        normalized = {}
        for key in ("host", "expected_hostname", "ib_addr", "socket_ifname", "ib_hca"):
            value = host.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`launcher.ssh.hosts[{rank}].{key}` must be a non-empty string")
            normalized[key] = value.strip()
        if re.fullmatch(r"[A-Za-z0-9._-]+", normalized["host"]) is None:
            raise ValueError(f"unsupported SSH host alias: {normalized['host']!r}")
        if normalized["host"] in seen_aliases:
            raise ValueError(f"duplicate SSH host alias: {normalized['host']!r}")
        seen_aliases.add(normalized["host"])
        try:
            ipaddress.IPv4Address(normalized["ib_addr"])
        except ipaddress.AddressValueError as exc:
            raise ValueError(f"invalid InfiniBand IPv4 address: {normalized['ib_addr']!r}") from exc
        for key in ("expected_hostname", "socket_ifname", "ib_hca"):
            if re.fullmatch(r"[A-Za-z0-9._-]+", normalized[key]) is None:
                raise ValueError(f"unsupported {key}: {normalized[key]!r}")
        normalized_hosts.append(normalized)
    ssh_hosts = normalized_hosts
    if ssh_hosts[0]["ib_addr"] != master_addr:
        raise ValueError("SSH rank 0 `ib_addr` must equal `launcher.master_addr`")

    rdma_preflight = require_mapping(ssh, "rdma_preflight")
    if rdma_preflight.get("enabled") is not True:
        raise ValueError("SSH mode requires `launcher.ssh.rdma_preflight.enabled: true`")
    for key in ("port", "nccl_port", "message_size", "iterations", "timeout_sec"):
        require_positive_int(rdma_preflight, key, "launcher.ssh.rdma_preflight")
    min_gbps = rdma_preflight.get("min_gbps")
    if isinstance(min_gbps, bool) or not isinstance(min_gbps, (int, float)) or min_gbps <= 0:
        raise ValueError("`launcher.ssh.rdma_preflight.min_gbps` must be positive")

runtime_training_config = copy.deepcopy(source_config)
runtime_training_config.pop("launcher")
runtime_training_config.pop("deepspeed")
runtime_training_config["run_id"] = run_id
runtime_training_config["run_root_dir"] = str(run_root_dir)


def materialize_repo_path(parent: dict, key: str) -> None:
    value = parent.get(key)
    if isinstance(value, str) and value.strip():
        parent[key] = str(resolve_repo_path(value, key))


# Public configs keep machine-specific resources under repository-relative
# local/. Materialize every supported training path before handing the config
# to the trainer, so execution is independent of the caller's working directory.
framework = runtime_training_config.get("framework", {})
qwenvl = framework.get("qwenvl", {}) if isinstance(framework, dict) else {}
navvla = framework.get("navvla", {}) if isinstance(framework, dict) else {}
if isinstance(qwenvl, dict):
    materialize_repo_path(qwenvl, "base_vlm")
if isinstance(navvla, dict):
    materialize_repo_path(navvla, "visual_cache_encoder_ckpt")
for dataset in runtime_training_config.get("datasets", {}).get("vla_data", {}).get("datasets", []):
    if isinstance(dataset, dict):
        materialize_repo_path(dataset, "data_root_dir")
openloop_eval = runtime_training_config.get("trainer", {}).get("openloop_eval", {})
if isinstance(openloop_eval, dict):
    materialize_repo_path(openloop_eval, "targets_root")
    for dataset in openloop_eval.get("datasets", []):
        if isinstance(dataset, dict):
            materialize_repo_path(dataset, "eval_root_dir")

deepspeed_path = runtime_dir / "deepspeed.generated.json"
accelerate_path = runtime_dir / "accelerate.generated.yaml"
training_config_path = runtime_dir / "config.launch.yaml"

accelerate_config = {
    "compute_environment": "LOCAL_MACHINE",
    "debug": False,
    "deepspeed_config": {
        "deepspeed_multinode_launcher": "standard",
        "zero3_init_flag": False,
    },
    "distributed_type": "DEEPSPEED",
    "num_machines": num_machines,
    "num_processes": num_processes,
    "rdzv_backend": "static",
    "same_network": True,
}

deepspeed_path.write_text(json.dumps(deepspeed, indent=2) + "\n", encoding="utf-8")
accelerate_path.write_text(
    yaml.safe_dump(accelerate_config, sort_keys=False),
    encoding="utf-8",
)
training_config_path.write_text(
    yaml.safe_dump(runtime_training_config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

metadata = {
    "LAUNCHER_MODE": mode,
    "NUM_MACHINES": num_machines,
    "NUM_PROCESSES": num_processes,
    "PROCESSES_PER_MACHINE": processes_per_machine,
    "MAIN_PROCESS_PORT": main_process_port,
    "MASTER_ADDR_CONFIG": master_addr,
    "CUDA_VISIBLE_DEVICES_CONFIG": cuda_visible_devices,
    "CUDA_HOME_CONFIG": launcher.get("cuda_home", ""),
    "PYTHON_BIN": str(python_bin),
    "TRAINING_SCRIPT": str(training_script),
    "TRAINING_CONFIG": str(training_config_path),
    "ACCELERATE_CONFIG": str(accelerate_path),
    "DEEPSPEED_CONFIG": str(deepspeed_path),
    "RUN_ID": run_id,
    "RUN_ROOT_DIR": str(run_root_dir),
    "OUTPUT_DIR": str(run_root_dir / run_id),
    "GLOBAL_BATCH_SIZE": global_batch_size,
}
if mode == "ssh":
    for rank, host in enumerate(ssh_hosts):
        for key, value in host.items():
            metadata[f"SSH_{key.upper()}_{rank}"] = value
    metadata.update(
        {
            "RDMA_PREFLIGHT_PORT": rdma_preflight["port"],
            "NCCL_PREFLIGHT_PORT": rdma_preflight["nccl_port"],
            "RDMA_PREFLIGHT_MESSAGE_SIZE": rdma_preflight["message_size"],
            "RDMA_PREFLIGHT_ITERATIONS": rdma_preflight["iterations"],
            "RDMA_PREFLIGHT_TIMEOUT_SEC": rdma_preflight["timeout_sec"],
            "RDMA_PREFLIGHT_MIN_GBPS": rdma_preflight["min_gbps"],
        }
    )
with (runtime_dir / "metadata.sh").open("w", encoding="utf-8") as stream:
    for key, value in metadata.items():
        stream.write(f"{key}={shlex.quote(str(value))}\n")

with (runtime_dir / "environment.sh").open("w", encoding="utf-8") as stream:
    for key, value in environment.items():
        if value is None:
            continue
        stream.write(f"export {key}={shlex.quote(str(value))}\n")

required_paths = [python_bin, training_script, config_path]
base_vlm = runtime_training_config.get("framework", {}).get("qwenvl", {}).get("base_vlm")
if isinstance(base_vlm, str) and base_vlm.strip():
    required_paths.append(Path(base_vlm).expanduser())
for dataset in runtime_training_config.get("datasets", {}).get("vla_data", {}).get("datasets", []):
    if not isinstance(dataset, dict):
        continue
    data_root = dataset.get("data_root_dir")
    if isinstance(data_root, str) and data_root.strip():
        required_paths.append(Path(data_root).expanduser())
with (runtime_dir / "required_paths.txt").open("w", encoding="utf-8") as stream:
    for path in dict.fromkeys(str(path) for path in required_paths):
        stream.write(path + "\n")
PY

# shellcheck disable=SC1091
source "${runtime_dir}/metadata.sh"
# shellcheck disable=SC1091
source "${runtime_dir}/environment.sh"

if [[ ! -x ${PYTHON_BIN} ]]; then
  echo "ERROR: launcher Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f ${TRAINING_SCRIPT} ]]; then
  echo "ERROR: training script not found: ${TRAINING_SCRIPT}" >&2
  exit 1
fi

machine_rank=0
master_addr=
ssh_worker=false
case ${LAUNCHER_MODE} in
  local)
    ;;
  slurm)
    : "${SLURM_NODEID:?SLURM_NODEID is required for launcher.mode=slurm}"
    if [[ ! ${SLURM_NODEID} =~ ^[0-9]+$ ]] || (( SLURM_NODEID >= NUM_MACHINES )); then
      echo "ERROR: invalid SLURM_NODEID=${SLURM_NODEID} for ${NUM_MACHINES} machines" >&2
      exit 1
    fi
    machine_rank=${SLURM_NODEID}
    if [[ ${MASTER_ADDR_CONFIG} != auto ]]; then
      master_addr=${MASTER_ADDR_CONFIG}
    else
      : "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required when launcher.master_addr=auto}"
      master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')
    fi
    if [[ -z ${master_addr} ]]; then
      echo "ERROR: unable to determine the main process address" >&2
      exit 1
    fi
    export MASTER_ADDR=${master_addr}
    export MASTER_PORT=${MAIN_PROCESS_PORT}
    ;;
  ssh)
    master_addr=${MASTER_ADDR_CONFIG}
    if [[ -n ${NAVVLA_MACHINE_RANK:-} ]]; then
      if [[ ! ${NAVVLA_MACHINE_RANK} =~ ^[01]$ ]]; then
        echo "ERROR: invalid NAVVLA_MACHINE_RANK=${NAVVLA_MACHINE_RANK}" >&2
        exit 1
      fi
      machine_rank=${NAVVLA_MACHINE_RANK}
      ssh_worker=true
      export MASTER_ADDR=${master_addr}
      export MASTER_PORT=${MAIN_PROCESS_PORT}

      host_var=SSH_HOST_${machine_rank}
      expected_hostname_var=SSH_EXPECTED_HOSTNAME_${machine_rank}
      ib_addr_var=SSH_IB_ADDR_${machine_rank}
      socket_ifname_var=SSH_SOCKET_IFNAME_${machine_rank}
      ib_hca_var=SSH_IB_HCA_${machine_rank}
      host=${!host_var}
      expected_hostname=${!expected_hostname_var}
      ib_addr=${!ib_addr_var}
      socket_ifname=${!socket_ifname_var}
      ib_hca=${!ib_hca_var}

      if [[ ${dry_run} == true ]]; then
        actual_hostname=${expected_hostname}
      else
        actual_hostname=$(hostname -s)
        if [[ ${actual_hostname} != "${expected_hostname}" ]]; then
          echo "ERROR: rank ${machine_rank} expected hostname ${expected_hostname}, got ${actual_hostname}" >&2
          exit 1
        fi
        if ! ip -brief address show "${socket_ifname}" | grep -Fq "${ib_addr}/"; then
          echo "ERROR: ${socket_ifname} does not own InfiniBand address ${ib_addr}" >&2
          exit 1
        fi
        if ! ibdev2netdev | grep -Eq "^${ib_hca} port 1 ==> ${socket_ifname} \(Up\)$"; then
          echo "ERROR: ${ib_hca} is not mapped to active netdev ${socket_ifname}" >&2
          exit 1
        fi
        ib_status=$(ibstat "${ib_hca}")
        if ! grep -Fq 'State: Active' <<<"${ib_status}" ||
           ! grep -Fq 'Physical state: LinkUp' <<<"${ib_status}" ||
           ! grep -Fq 'Rate: 200' <<<"${ib_status}"; then
          echo "ERROR: ${ib_hca} is not Active/LinkUp at 200Gb" >&2
          exit 1
        fi
        while IFS= read -r required_path; do
          if [[ ! -e ${required_path} ]]; then
            echo "ERROR: required shared path missing on ${actual_hostname}: ${required_path}" >&2
            exit 1
          fi
        done < "${runtime_dir}/required_paths.txt"
      fi

      export NCCL_SOCKET_IFNAME="=${socket_ifname}"
      export GLOO_SOCKET_IFNAME=${socket_ifname}
      export NCCL_IB_HCA=${ib_hca}
      echo "IB_BINDING rank=${machine_rank} host=${actual_hostname} ib_addr=${ib_addr} socket=${NCCL_SOCKET_IFNAME} gloo=${GLOO_SOCKET_IFNAME} hca=${NCCL_IB_HCA} NCCL_NET=${NCCL_NET} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
    fi
    ;;
esac

export PYTHONPATH="${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
if [[ ${CUDA_VISIBLE_DEVICES_CONFIG} != all ]]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_CONFIG}
fi

launch_command=(
  "${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch
  --main_process_port "${MAIN_PROCESS_PORT}"
  --config_file "${ACCELERATE_CONFIG}"
  --deepspeed_config_file "${DEEPSPEED_CONFIG}"
  --num_processes "${NUM_PROCESSES}"
)
if [[ ${LAUNCHER_MODE} == slurm || ( ${LAUNCHER_MODE} == ssh && ${ssh_worker} == true ) ]]; then
  launch_command+=(
    --main_process_ip "${master_addr}"
    --machine_rank "${machine_rank}"
    --num_machines "${NUM_MACHINES}"
  )
fi
launch_command+=(
  "${TRAINING_SCRIPT}"
  --config_yaml "${TRAINING_CONFIG}"
)

echo "config: ${config_yaml}"
echo "mode: ${LAUNCHER_MODE}, machines: ${NUM_MACHINES}, processes: ${NUM_PROCESSES}"
echo "run_id: ${RUN_ID}"
echo "output_dir: ${OUTPUT_DIR}"
echo "global_batch_size: ${GLOBAL_BATCH_SIZE}"
printf 'command:'
printf ' %q' "${launch_command[@]}"
printf '\n'

if [[ ${dry_run} == true ]]; then
  if [[ ${LAUNCHER_MODE} == ssh && ${ssh_worker} == false ]]; then
    for dry_rank in 0 1; do
      NAVVLA_RUN_ID_OVERRIDE=${RUN_ID} NAVVLA_MACHINE_RANK=${dry_rank} \
        bash "${script_dir}/run_train.sh" "${config_yaml}" --dry-run
    done
  fi
  echo "dry-run complete"
  exit 0
fi

run_remote() {
  local host_alias=$1
  shift
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${host_alias}" "$@"
}

run_rdma_preflight() {
  local preflight_dir server_pid server_status client_status average_gbps
  preflight_dir=$(mktemp -d "${TMPDIR:-/tmp}/navvla-rdma.XXXXXX")
  run_remote "${SSH_HOST_0}" \
    "timeout ${RDMA_PREFLIGHT_TIMEOUT_SEC}s ib_write_bw -d ${SSH_IB_HCA_0} -p ${RDMA_PREFLIGHT_PORT} -s ${RDMA_PREFLIGHT_MESSAGE_SIZE} -n ${RDMA_PREFLIGHT_ITERATIONS} --report_gbits" \
    >"${preflight_dir}/server.log" 2>&1 &
  server_pid=$!
  sleep 1
  set +e
  run_remote "${SSH_HOST_1}" \
    "timeout ${RDMA_PREFLIGHT_TIMEOUT_SEC}s ib_write_bw ${SSH_IB_ADDR_0} -d ${SSH_IB_HCA_1} -p ${RDMA_PREFLIGHT_PORT} -s ${RDMA_PREFLIGHT_MESSAGE_SIZE} -n ${RDMA_PREFLIGHT_ITERATIONS} --report_gbits" \
    >"${preflight_dir}/client.log" 2>&1
  client_status=$?
  wait "${server_pid}"
  server_status=$?
  set -e
  if (( server_status != 0 || client_status != 0 )); then
    echo "ERROR: InfiniBand RDMA bandwidth preflight failed" >&2
    tail -80 "${preflight_dir}/server.log" >&2 || true
    tail -80 "${preflight_dir}/client.log" >&2 || true
    return 1
  fi
  if ! grep -Fq 'Transport type : IB' "${preflight_dir}/server.log" ||
     ! grep -Fq 'Link type       : IB' "${preflight_dir}/server.log"; then
    echo "ERROR: ib_write_bw did not report an InfiniBand transport" >&2
    return 1
  fi
  average_gbps=$(awk '$1 ~ /^[0-9]+$/ && NF >= 4 {value=$4} END {print value}' "${preflight_dir}/client.log")
  if [[ -z ${average_gbps} ]] ||
     ! awk -v actual="${average_gbps}" -v minimum="${RDMA_PREFLIGHT_MIN_GBPS}" 'BEGIN {exit !(actual >= minimum)}'; then
    echo "ERROR: InfiniBand bandwidth ${average_gbps:-unknown} Gb/s is below required ${RDMA_PREFLIGHT_MIN_GBPS} Gb/s" >&2
    return 1
  fi
  echo "IB_WRITE_BW_OK server=${SSH_HOST_0}/${SSH_IB_HCA_0} client=${SSH_HOST_1}/${SSH_IB_HCA_1} average_gbps=${average_gbps}"
}

run_nccl_preflight() {
  local preflight_dir smoke_program quoted_program common_env pid0 pid1 status0 status1
  preflight_dir=$(mktemp -d "${TMPDIR:-/tmp}/navvla-nccl.XXXXXX")
  smoke_program='import torch, torch.distributed as dist; torch.cuda.set_device(0); dist.init_process_group("nccl", init_method="env://"); x=torch.tensor([float(dist.get_rank()+1)], device="cuda"); dist.all_reduce(x); torch.cuda.synchronize(); print(f"NCCL_SMOKE rank={dist.get_rank()} sum={x.item()} backend={dist.get_backend()}", flush=True); dist.destroy_process_group()'
  quoted_program=$(printf '%q' "${smoke_program}")
  common_env="MASTER_ADDR=${SSH_IB_ADDR_0} MASTER_PORT=${NCCL_PREFLIGHT_PORT} WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0 NCCL_IB_DISABLE=0 NCCL_NET=IB NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET PYTHONNOUSERSITE=1"
  run_remote "${SSH_HOST_0}" \
    "cd $(printf '%q' "${code_root}") && env ${common_env} RANK=0 LOCAL_RANK=0 NCCL_SOCKET_IFNAME==${SSH_SOCKET_IFNAME_0} GLOO_SOCKET_IFNAME=${SSH_SOCKET_IFNAME_0} NCCL_IB_HCA=${SSH_IB_HCA_0} $(printf '%q' "${PYTHON_BIN}") -c ${quoted_program}" \
    >"${preflight_dir}/rank0.log" 2>&1 &
  pid0=$!
  run_remote "${SSH_HOST_1}" \
    "cd $(printf '%q' "${code_root}") && env ${common_env} RANK=1 LOCAL_RANK=0 NCCL_SOCKET_IFNAME==${SSH_SOCKET_IFNAME_1} GLOO_SOCKET_IFNAME=${SSH_SOCKET_IFNAME_1} NCCL_IB_HCA=${SSH_IB_HCA_1} $(printf '%q' "${PYTHON_BIN}") -c ${quoted_program}" \
    >"${preflight_dir}/rank1.log" 2>&1 &
  pid1=$!
  set +e
  wait "${pid0}"; status0=$?
  wait "${pid1}"; status1=$?
  set -e
  if (( status0 != 0 || status1 != 0 )) ||
     ! grep -Fq 'NET/IB' "${preflight_dir}/rank0.log" ||
     ! grep -Fq 'NET/IB' "${preflight_dir}/rank1.log" ||
     ! grep -Fq 'NCCL_SMOKE rank=0 sum=3.0 backend=nccl' "${preflight_dir}/rank0.log" ||
     ! grep -Fq 'NCCL_SMOKE rank=1 sum=3.0 backend=nccl' "${preflight_dir}/rank1.log"; then
    echo "ERROR: NCCL InfiniBand all-reduce preflight failed" >&2
    tail -100 "${preflight_dir}/rank0.log" >&2 || true
    tail -100 "${preflight_dir}/rank1.log" >&2 || true
    return 1
  fi
  if grep -Fq 'NET/Socket' "${preflight_dir}/rank0.log" || grep -Fq 'NET/Socket' "${preflight_dir}/rank1.log"; then
    echo "ERROR: NCCL selected socket transport despite NCCL_NET=IB" >&2
    return 1
  fi
  echo "NCCL_IB_OK rank0=${SSH_HOST_0}/${SSH_IB_HCA_0} rank1=${SSH_HOST_1}/${SSH_IB_HCA_1} transport=NET/IB sum=3.0"
}

run_ssh_host_preflight() {
  local rank host_var expected_hostname_var ib_addr_var socket_ifname_var ib_hca_var host expected_hostname ib_addr socket_ifname ib_hca remote_script gpu_processes
  for rank in 0 1; do
    host_var=SSH_HOST_${rank}; expected_hostname_var=SSH_EXPECTED_HOSTNAME_${rank}; ib_addr_var=SSH_IB_ADDR_${rank}; socket_ifname_var=SSH_SOCKET_IFNAME_${rank}; ib_hca_var=SSH_IB_HCA_${rank}
    host=${!host_var}; expected_hostname=${!expected_hostname_var}; ib_addr=${!ib_addr_var}; socket_ifname=${!socket_ifname_var}; ib_hca=${!ib_hca_var}
    remote_script="set -euo pipefail; test \"\$(hostname -s)\" = $(printf '%q' "${expected_hostname}"); ip -brief address show $(printf '%q' "${socket_ifname}") | grep -F $(printf '%q' "${ib_addr}/"); ibdev2netdev | grep -E $(printf '%q' "^${ib_hca} port 1 ==> ${socket_ifname} \\(Up\\)\$"); ibstat $(printf '%q' "${ib_hca}") | grep -F 'State: Active'; ibstat $(printf '%q' "${ib_hca}") | grep -F 'Physical state: LinkUp'; ibstat $(printf '%q' "${ib_hca}") | grep -F 'Rate: 200'; command -v ib_write_bw >/dev/null"
    while IFS= read -r required_path; do
      remote_script+="; test -e $(printf '%q' "${required_path}")"
    done < "${runtime_dir}/required_paths.txt"
    if (( rank == 0 )); then
      remote_script+="; ! ss -ltn | grep -E $(printf '%q' ":${MAIN_PROCESS_PORT}[[:space:]]")"
    fi
    run_remote "${host}" "${remote_script}" >/dev/null
    if ! gpu_processes=$(run_remote "${host}" "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader"); then
      echo "ERROR: unable to query GPU compute processes on ${host}" >&2
      return 1
    fi
    if [[ -n ${gpu_processes} ]]; then
      echo "ERROR: GPU compute processes are already running on ${host}; refusing to run NCCL smoke or training" >&2
      printf '%s\n' "${gpu_processes}" >&2
      return 1
    fi
    echo "IB_HOST_OK rank=${rank} host=${host} expected_hostname=${expected_hostname} ib_addr=${ib_addr} socket=${socket_ifname} hca=${ib_hca} rate_gbps=200"
  done
  run_remote "${SSH_HOST_1}" "ping -I $(printf '%q' "${SSH_SOCKET_IFNAME_1}") -c 3 -W 2 $(printf '%q' "${SSH_IB_ADDR_0}")" >/dev/null
  run_remote "${SSH_HOST_0}" "ping -I $(printf '%q' "${SSH_SOCKET_IFNAME_0}") -c 3 -W 2 $(printf '%q' "${SSH_IB_ADDR_1}")" >/dev/null
  echo "IB_PING_OK ${SSH_HOST_0}:${SSH_IB_ADDR_0}<->${SSH_HOST_1}:${SSH_IB_ADDR_1}"
  run_rdma_preflight
  run_nccl_preflight
}

if [[ ${LAUNCHER_MODE} == ssh && ${ssh_worker} == false ]]; then
  run_ssh_host_preflight
  if [[ ${preflight_only} == true ]]; then
    echo "preflight complete"
    exit 0
  fi

  worker_env=("NAVVLA_RUN_ID_OVERRIDE=${RUN_ID}")
  remote_env=$(printf '%q ' "${worker_env[@]}")
  remote_code_root=$(printf '%q' "${code_root}")
  remote_entrypoint=$(printf '%q' "${code_root}/examples/NavVLA/train_files/qwen35/run_train.sh")
  remote_config=$(printf '%q' "${config_yaml}")
  child_pids=()
  for machine_rank in 0 1; do
    host_var=SSH_HOST_${machine_rank}
    host=${!host_var}
    run_remote "${host}" "cd ${remote_code_root} && ${remote_env} NAVVLA_MACHINE_RANK=${machine_rank} bash ${remote_entrypoint} ${remote_config}" &
    child_pids+=("$!")
  done
  cleanup_children() {
    local pid
    for pid in "${child_pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then kill "${pid}" 2>/dev/null || true; fi
    done
    for pid in "${child_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  }
  trap 'cleanup_children; exit 130' INT
  trap 'cleanup_children; exit 143' TERM
  remaining_children=${#child_pids[@]}
  while (( remaining_children > 0 )); do
    if wait -n; then
      remaining_children=$((remaining_children - 1))
    else
      child_status=$?
      trap - INT TERM
      cleanup_children
      exit "${child_status}"
    fi
  done
  trap - INT TERM
  exit 0
fi

if [[ ${preflight_only} == true ]]; then
  echo "ERROR: --preflight-only is only valid for top-level SSH mode" >&2
  exit 2
fi

if [[ -n ${CUDA_HOME_CONFIG} ]]; then
  if [[ ${CUDA_HOME_CONFIG} = /* ]]; then
    export CUDA_HOME=${CUDA_HOME_CONFIG}
  else
    export CUDA_HOME=${code_root}/${CUDA_HOME_CONFIG}
  fi
fi
if [[ ! -x ${CUDA_HOME:-}/bin/nvcc ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  else
    echo "ERROR: CUDA_HOME does not contain bin/nvcc and /usr/local/cuda is unavailable." >&2
    exit 1
  fi
fi
export PATH="${CUDA_HOME}/bin:${PATH}"

# DeepSpeed/Triton file locks are unreliable on the NFS-backed home directory.
triton_cache_dir=${TRITON_CACHE_DIR:-/dev/shm/$(id -un)/triton_cache}
mkdir -p "${triton_cache_dir}"
export TRITON_CACHE_DIR=${triton_cache_dir}

if (( machine_rank == 0 )); then
  mkdir -p "${OUTPUT_DIR}"
  cp "${config_yaml}" "${OUTPUT_DIR}/config.source.yaml"
  cp "${TRAINING_CONFIG}" "${OUTPUT_DIR}/config.launch.yaml"
  cp "${ACCELERATE_CONFIG}" "${OUTPUT_DIR}/accelerate.generated.yaml"
  cp "${DEEPSPEED_CONFIG}" "${OUTPUT_DIR}/deepspeed.generated.json"
  cp "${script_dir}/run_train.sh" "${OUTPUT_DIR}/"
fi

cd "${code_root}"
"${launch_command[@]}"
