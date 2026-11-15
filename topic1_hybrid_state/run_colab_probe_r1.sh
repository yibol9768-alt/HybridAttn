#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/liuyibo/auto-llm-driving/HybridAttn/topic1_hybrid_state

echo "=== [1/5] Verifying Frozen Preregistration SHA-256 ==="
sha256sum -c PREREG_R1.sha256

cleanup() {
  echo "=== [5/5] Stopping Colab Session probe-r1 & Verifying Zero Burn ==="
  colab stop -s probe-r1 || true
  colab usage || true
}
trap cleanup EXIT

echo "=== [2/5] Allocating Google Colab L4 GPU Session from LinuxDesk ==="
colab new -s probe-r1 --gpu L4
colab status -s probe-r1

echo "=== [3/5] Uploading Audit-Hardened Probe Script & Executing (with 2400s Timeout) ==="
colab upload -s probe-r1 colab_probe_r1.py /content/colab_probe_r1.py

HF_TOK="$(cat "$HOME/.cache/huggingface/token")"
colab exec -s probe-r1 --timeout 2400 --env "HF_TOKEN=${HF_TOK}" << "PY"
import shutil, subprocess, sys

print("[vm-setup] Installing locked transformers commit...", flush=True)
uv_bin = shutil.which("uv")
pkgs = [
    "https://github.com/huggingface/transformers/archive/0f7a8768733c448751f25abcb4b6b3ffef829cf5.tar.gz",
    "accelerate",
    "safetensors",
]
cmd = [uv_bin, "pip", "install", "--system", *pkgs] if uv_bin else [sys.executable, "-m", "pip", "install", "-q", *pkgs]
subprocess.run(cmd, check=True)
print("[vm-setup] Dependencies installed cleanly.", flush=True)

proc = subprocess.Popen(
    [sys.executable, "-u", "/content/colab_probe_r1.py", "--deps-ready", "--out-dir", "/content/probe_r1_out", "--budget-seconds", "1200"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
if rc != 0:
    raise SystemExit(rc)
PY

echo "=== [4/5] Downloading Results Archive & Session Notebook Provenance ==="
mkdir -p results_r1
colab download -s probe-r1 /content/probe_r1_results_light.tar.gz results_r1/probe_r1_results_light.tar.gz
tar -xzf results_r1/probe_r1_results_light.tar.gz -C results_r1/
colab log -s probe-r1 -o results_r1/session_probe_r1.ipynb || true
ls -la results_r1/probe_r1_out/
