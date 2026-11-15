#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
if ! command -v timeout >/dev/null 2>&1; then
  timeout() { shift; "$@"; }
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SESSION_NAME="bench-r22"
REMOTE_DIR="/content"
REMOTE_OUT="/content/bench_r22_out"
LOCAL_RESULTS_DIR="${SCRIPT_DIR}/results_r22"
MODELS_ORDER=("0.8B" "2B" "4B" "9B")

verify_hashes() {
  echo "=== [$(date +"%Y-%m-%dT%H:%M:%S%z")] [1] Verifying Frozen Preregistration SHA-256 (R1, R2, R22) ==="
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c PREREG_R1.sha256
    sha256sum -c PREREG_R2.sha256
    sha256sum -c PREREG_R22.sha256
  else
    shasum -a 256 -c PREREG_R1.sha256
    shasum -a 256 -c PREREG_R2.sha256
    shasum -a 256 -c PREREG_R22.sha256
  fi
}

retry_cmd() {
  local max_tries="$1"
  local sleep_s="$2"
  shift 2
  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi
    if (( attempt >= max_tries )); then
      echo "ERROR: Command failed after ${max_tries} attempts: $*" >&2
      return 1
    fi
    echo "[retry ${attempt}/${max_tries}] Transient failure; retrying in ${sleep_s}s..." >&2
    sleep "${sleep_s}"
    attempt=$((attempt + 1))
  done
}

stop_bench_session() {
  echo "=== [$(date +"%Y-%m-%dT%H:%M:%S%z")] [Cleanup] Stopping ONLY Colab session ${SESSION_NAME} (never touching other sessions) ==="
  for _ in 1 2 3; do
    timeout 60 colab stop -s "${SESSION_NAME}" && break || sleep 5
  done
  timeout 45 colab sessions || true
}

upload_and_verify_vm() {
  verify_hashes
  echo "=== [$(date +"%Y-%m-%dT%H:%M:%S%z")] [2] Checking / Allocating Colab Session (-s ${SESSION_NAME}) ==="
  status_out="$(timeout 60 colab status -s "${SESSION_NAME}" 2>&1 || true)"
  if echo "${status_out}" | grep -qiE "not found|lost|error"; then
    echo "=== [$(date +"%Y-%m-%dT%H:%M:%S%z")] [2b] Allocating Single Google Colab A100 Session (-s ${SESSION_NAME}) ==="
    retry_cmd 5 10 timeout 180 colab new -s "${SESSION_NAME}" --gpu A100 < /dev/null
  fi
  timeout 60 colab status -s "${SESSION_NAME}"

  echo "=== [$(date +"%Y-%m-%dT%H:%M:%S%z")] [3] Uploading Frozen Preregistration Artifacts to ${SESSION_NAME}:${REMOTE_DIR} ==="
  for f in PREREG_R1.sha256 PREREG_R2.sha256 PREREG_R22.sha256 METHOD_AND_THEORY_R22.md smoke_r22.py SMOKE_R22_RESULTS.json colab_bench_r2.py colab_bench_r22.py; do
    retry_cmd 5 8 timeout 120 colab upload -s "${SESSION_NAME}" "${f}" "${REMOTE_DIR}/${f}"
  done
  timeout 90 colab exec -s "${SESSION_NAME}" --timeout 60 << 'PY'
from pathlib import Path
Path('/content/research_sources/qwen35_08b_inst').mkdir(parents=True, exist_ok=True)
PY
  retry_cmd 5 8 timeout 120 colab upload -s "${SESSION_NAME}" "research_sources/qwen35_08b_inst/tokenizer.json" "/content/research_sources/qwen35_08b_inst/tokenizer.json"

  if [[ -f "/tmp/r2_mech_dev.tar.gz" ]]; then
    retry_cmd 5 8 timeout 120 colab upload -s "${SESSION_NAME}" "/tmp/r2_mech_dev.tar.gz" "/content/r2_mech_dev.tar.gz"
    timeout 90 colab exec -s "${SESSION_NAME}" --timeout 60 << 'PY'
import tarfile
tarfile.open('/content/r2_mech_dev.tar.gz').extractall('/content')
PY
  fi

  echo "=== [$(date +"%Y-%m-%dT%H:%M:%S%z")] [4] Installing Locked transformers Commit & Running Full PyTorch Tiny-Arch Self-Test + Prereg Check on ${SESSION_NAME} ==="
  setup_vm_deps_and_selftest
}

_do_vm_setup() {
  HF_TOK="$(cat "$HOME/.cache/huggingface/token")"
  timeout 1800 colab exec -s "${SESSION_NAME}" --timeout 1800 --env "HF_TOKEN=${HF_TOK}" --env "USE_HUB_KERNELS=OFF" << 'PY'
import os, shutil, subprocess, sys
os.environ["USE_HUB_KERNELS"] = "OFF"
print("[vm-setup-r22] Installing locked transformers commit 0f7a8768733c448751f25abcb4b6b3ffef829cf5...", flush=True)
uv_bin = shutil.which("uv")
pkgs = [
    "https://github.com/huggingface/transformers/archive/0f7a8768733c448751f25abcb4b6b3ffef829cf5.tar.gz",
    "accelerate",
    "safetensors",
]
cmd = [uv_bin, "pip", "install", "--system", *pkgs] if uv_bin else [sys.executable, "-m", "pip", "install", "-q", *pkgs]
subprocess.run(cmd, check=True)

import torch
assert torch.cuda.is_available(), "CUDA unavailable on bench-r22"
props = torch.cuda.get_device_properties(0)
uuid_str = str(getattr(props, "uuid", "")).strip()
print(f"[vm-setup-r22] GPU={props.name}, VRAM={props.total_memory / (2**30):.2f} GiB, UUID={uuid_str}", flush=True)
assert "A100" in props.name and props.total_memory >= 38 * (2**30), f"Expected A100 >=38GiB, got {props.name}"
assert uuid_str, "Empty CUDA UUID"

subprocess.run([sys.executable, "-u", "/content/colab_bench_r22.py", "--verify-prereg", "/content/PREREG_R22.sha256", "--cpu-self-test"], check=True)
print("[vm-setup-r22] A100 hardware, PyTorch tiny-arch self-test, smoke_r22, and PREREG_R22.sha256 verified.", flush=True)
PY
}

setup_vm_deps_and_selftest() {
  retry_cmd 5 8 _do_vm_setup
}

pack_remote_model_tar() {
  local model_tag="$1"
  timeout 180 colab exec -s "${SESSION_NAME}" --timeout 120 --env "TARGET_MODEL=${model_tag}" << 'PY'
import os, tarfile
from pathlib import Path

tag = os.environ["TARGET_MODEL"]
out = Path("/content/bench_r22_out")
tar_path = Path(f"/content/pull_r22_{tag}.tar.gz")
with tarfile.open(tar_path, "w:gz") as tf:
    mdir = out / f"model_{tag}"
    if mdir.exists():
        tf.add(mdir, arcname=f"bench_r22_out/model_{tag}")
    for top in ("session_hardware_lock.json", "summary_r22.json"):
        p = out / top
        if p.exists():
            tf.add(p, arcname=f"bench_r22_out/{top}")
print(f"[pull-r22] Created {tar_path} ({tar_path.stat().st_size / 1024:.1f} KiB)", flush=True)
PY
}

pull_model_outputs() {
  local model_tag="$1"
  mkdir -p "${LOCAL_RESULTS_DIR}/bench_r22_out"
  echo "=== [$(date +"%Y-%m-%dT%H:%M:%S%z")] [Pull] Archiving & pulling outputs after model ${model_tag} from ${SESSION_NAME} ==="
  retry_cmd 6 10 pack_remote_model_tar "${model_tag}"
  retry_cmd 6 10 timeout 240 colab download -s "${SESSION_NAME}" "/content/pull_r22_${model_tag}.tar.gz" "${LOCAL_RESULTS_DIR}/pull_r22_${model_tag}.tar.gz"
  tar -xzf "${LOCAL_RESULTS_DIR}/pull_r22_${model_tag}.tar.gz" -C "${LOCAL_RESULTS_DIR}/"
  echo "=== [$(date +"%Y-%m-%dT%H:%M:%S%z")] [Pull] Extracted ${model_tag} into ${LOCAL_RESULTS_DIR}/bench_r22_out ==="
}

launch_remote_detached_model() {
  local model_tag="$1"
  local hf_tok="$2"
  timeout 120 colab exec -s "${SESSION_NAME}" --timeout 60 --env "HF_TOKEN=${hf_tok}" --env "USE_HUB_KERNELS=OFF" --env "RUN_MODEL=${model_tag}" << 'PY'
import json, os, subprocess, sys
from pathlib import Path

tag = os.environ["RUN_MODEL"]
summary_p = Path(f"/content/bench_r22_out/model_{tag}/summary_model_r22.json")
if summary_p.is_file():
    sm = json.loads(summary_p.read_text())
    if sm.get("status") == "COMPLETE":
        print(f"[remote-launch-r22] Model {tag} ALREADY COMPLETE on VM; skipping re-run.", flush=True)
        Path(f"/content/run_r22_{tag}.rc").write_text("0\n")
        raise SystemExit(0)

pid_file = Path(f"/content/run_r22_{tag}.pid")
rc_file = Path(f"/content/run_r22_{tag}.rc")
log_file = Path(f"/content/run_r22_{tag}.log")

if pid_file.is_file() and not rc_file.is_file():
    old_pid = int(pid_file.read_text().strip())
    stat_p = Path(f"/proc/{old_pid}/stat")
    if stat_p.is_file() and " Z " not in stat_p.read_text():
        print(f"[remote-launch-r22] Model {tag} already actively running at PID {old_pid}.", flush=True)
        raise SystemExit(0)

rc_file.unlink(missing_ok=True)
log_file.write_text("")
env = dict(os.environ)
env["USE_HUB_KERNELS"] = "OFF"
runner_code = f"""
import subprocess, sys
from pathlib import Path
log_f = open({str(log_file)!r}, "a", buffering=1)
cmd = [
    sys.executable, "-u", "/content/colab_bench_r22.py",
    "--verify-prereg", "/content/PREREG_R22.sha256",
    "--out-dir", "/content/bench_r22_out",
    "--models", {tag!r},
]
rc = subprocess.call(cmd, stdout=log_f, stderr=subprocess.STDOUT)
log_f.flush()
log_f.close()
Path({str(rc_file)!r}).write_text(f"{{rc}}\\n")
"""
proc = subprocess.Popen(
    [sys.executable, "-u", "-c", runner_code],
    env=env,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
pid_file.write_text(f"{proc.pid}\n")
print(f"[remote-launch-r22] Started detached GPU worker for {tag} (PID={proc.pid})", flush=True)
PY
}

poll_remote_model_once() {
  local model_tag="$1"
  local offset="$2"
  timeout 60 colab exec -s "${SESSION_NAME}" --timeout 30 --env "RUN_MODEL=${model_tag}" --env "LOG_OFFSET=${offset}" << 'PY'
import os
from pathlib import Path

tag = os.environ["RUN_MODEL"]
offset = int(os.environ["LOG_OFFSET"])
log_file = Path(f"/content/run_r22_{tag}.log")
rc_file = Path(f"/content/run_r22_{tag}.rc")
pid_file = Path(f"/content/run_r22_{tag}.pid")

data = log_file.read_bytes() if log_file.is_file() else b""
new_bytes = data[offset:]
new_offset = len(data)
if new_bytes:
    print(new_bytes.decode("utf-8", errors="replace"), end="", flush=True)

if rc_file.is_file():
    rc_val = int(rc_file.read_text().strip())
    print(f"\n__COLAB_POLL_STATUS__:DONE:{rc_val}:{new_offset}", flush=True)
else:
    alive = False
    if pid_file.is_file():
        try:
            os.kill(int(pid_file.read_text().strip()), 0)
            alive = True
        except OSError:
            alive = False
    if alive:
        print(f"\n__COLAB_POLL_STATUS__:RUNNING:0:{new_offset}", flush=True)
    else:
        print(f"\n__COLAB_POLL_STATUS__:DEAD:1:{new_offset}", flush=True)
PY
}

run_single_model() {
  local model_tag="$1"
  verify_hashes
  echo "=== [$(date +"%Y-%m-%dT%H:%M:%S%z")] [Run] Executing model ${model_tag} on ${SESSION_NAME} ==="
  HF_TOK="$(cat "$HOME/.cache/huggingface/token")"
  retry_cmd 6 10 launch_remote_detached_model "${model_tag}" "${HF_TOK}"

  local offset=0
  local fail_streak=0
  local final_rc=0
  while true; do
    set +e
    out="$(poll_remote_model_once "${model_tag}" "${offset}" 2>&1)"
    poll_rc=$?
    set -e
    if [[ ${poll_rc} -ne 0 ]]; then
      fail_streak=$((fail_streak + 1))
      echo "[poll-warn ${fail_streak}] Transient poll error on ${model_tag}; GPU worker continues on VM. Sleeping 15s..."
      if (( fail_streak >= 20 )); then
        echo "ERROR: 20 consecutive poll failures for ${model_tag}" >&2
        return 1
      fi
      sleep 15
      continue
    fi
    fail_streak=0
    status_line="$(printf "%s\n" "${out}" | grep "^__COLAB_POLL_STATUS__:" | tail -n 1 || true)"
    clean_out="$(printf "%s\n" "${out}" | grep -v "^__COLAB_POLL_STATUS__:" || true)"
    if [[ -n "${clean_out}" ]]; then
      printf "%s\n" "${clean_out}"
    fi
    if [[ -n "${status_line}" ]]; then
      IFS=":" read -r _ state rc_val new_off <<< "${status_line}"
      offset="${new_off}"
      if [[ "${state}" == "DONE" ]]; then
        final_rc="${rc_val}"
        break
      elif [[ "${state}" == "DEAD" ]]; then
        echo "ERROR: Remote worker for ${model_tag} died without writing rc file" >&2
        final_rc=1
        break
      fi
    fi
    sleep 15
  done

  pull_model_outputs "${model_tag}" || true
  if [[ "${final_rc}" -ne 0 ]]; then
    echo "ERROR: Model ${model_tag} exited with code ${final_rc}" >&2
    return "${final_rc}"
  fi
}

pack_remote_all() {
  timeout 240 colab exec -s "${SESSION_NAME}" --timeout 180 << 'PY'
import subprocess, sys, tarfile
from pathlib import Path

subprocess.run([
    sys.executable, "-u", "/content/colab_bench_r22.py",
    "--verify-prereg", "/content/PREREG_R22.sha256",
    "--aggregate-only",
    "--out-dir", "/content/bench_r22_out",
], check=True)

tar_path = Path("/content/bench_r22_results_all.tar.gz")
with tarfile.open(tar_path, "w:gz") as tf:
    tf.add("/content/bench_r22_out", arcname="bench_r22_out")
print(f"[finalize-r22] Packed {tar_path} ({tar_path.stat().st_size / (1024*1024):.2f} MiB)", flush=True)
PY
}

finalize_and_pull_all() {
  mkdir -p "${LOCAL_RESULTS_DIR}/bench_r22_out"
  echo "=== [$(date +"%Y-%m-%dT%H:%M:%S%z")] [Finalize] Running cross-model aggregation & pulling complete archive + provenance notebook ==="
  retry_cmd 6 10 pack_remote_all
  retry_cmd 6 10 timeout 300 colab download -s "${SESSION_NAME}" "/content/bench_r22_results_all.tar.gz" "${LOCAL_RESULTS_DIR}/bench_r22_results_all.tar.gz"
  tar -xzf "${LOCAL_RESULTS_DIR}/bench_r22_results_all.tar.gz" -C "${LOCAL_RESULTS_DIR}/"
  timeout 120 colab log -s "${SESSION_NAME}" -o "${LOCAL_RESULTS_DIR}/session_bench_r22.ipynb" || true
}

MODE="${1:-setup}"
case "${MODE}" in
  all)
    upload_and_verify_vm
    for m in "${MODELS_ORDER[@]}"; do
      run_single_model "${m}"
    done
    finalize_and_pull_all
    stop_bench_session
    ;;
  resume)
    verify_hashes
    for m in "${MODELS_ORDER[@]}"; do
      run_single_model "${m}"
    done
    finalize_and_pull_all
    stop_bench_session
    ;;
  setup)
    upload_and_verify_vm
    ;;
  model)
    run_single_model "${2:?Usage: $0 model <TAG>}"
    ;;
  finalize)
    finalize_and_pull_all
    ;;
  stop)
    stop_bench_session
    ;;
  *)
    echo "Usage: $0 [all|setup|model <TAG>|finalize|stop]" >&2
    exit 2
    ;;
esac
