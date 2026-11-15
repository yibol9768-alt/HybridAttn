#!/usr/bin/env python3
"""Pre-registered Real-Weight GPU Probe (R1.1, Audit-Hardened) for Qwen3.5-0.8B-Base & 2B-Base.

Implements COLAB_PROBE_PLAN.md (R1.1) line-by-line:
- Disables external hub/FLA kernels (`USE_HUB_KERNELS=OFF`) BEFORE importing transformers so
  reference FP32 recurrence wrappers fire deterministically and call counts are strictly verified.
- Enforces single-token label identity (" alpha" vs " beta") + filler purity + dual tokenization log.
- Runs 6-arm causal cache intervention (O, K, G, B, C_pos, N_compact) separating GDN ghost+conv (c_G),
  FullAttn KV eviction (c_F), suffix KV content staleness (c_staleKV), and RoPE re-indexing (c_rope).
- Evaluates P1 (spectral duality & recovery horizon), P2 (uncensored closed-form oblique Theta via
  forward rank-1 biorthogonal recurrence + SVD cross-check), P3 (3-token conv1d halo scrub + 3-layer
  operand-defect certificate), P4 (isotropic spherical Monte Carlo control), and P5 (paired bootstrap).
"""

from __future__ import annotations

import os

os.environ["USE_HUB_KERNELS"] = "OFF"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

TRANSFORMERS_COMMIT = "0f7a8768733c448751f25abcb4b6b3ffef829cf5"
TRANSFORMERS_TARBALL = f"https://github.com/huggingface/transformers/archive/{TRANSFORMERS_COMMIT}.tar.gz"

MODELS = [
    ("0.8B", "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
    ("2B", "Qwen/Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"),
]

SEED = 20261109
CAPTURED_LAYERS = [0, 1, 2, 20, 21, 22]
CAPTURED_HEADS = [0, 5, 10, 15]
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23]
GDN_LAYERS = [i for i in range(24) if i % 4 != 3]
SPAN_START, SPAN_END = 128, 159  # 0-based inclusive, 32 tokens
BODY_LEN = 512
EPS = 1e-6


def ensure_transformers_installed(allow_install: bool) -> None:
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qm  # noqa: F401
        src_bytes = Path(qm.__file__).read_bytes()
        sha = hashlib.sha256(src_bytes).hexdigest()
        if sha == "4463b8a108f678c166755101d95f188179f54143305953d78147de0389b80ebc":
            return
        print(f"[setup] modeling_qwen3_5.py sha256={sha[:12]} differs from locked commit; reinstalling...", flush=True)
    except Exception as exc:
        print(f"[setup] Qwen3_5 import check ({exc}); installing locked transformers commit...", flush=True)

    if not allow_install:
        raise RuntimeError("Locked transformers commit not active even after --deps-ready")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", TRANSFORMERS_TARBALL, "accelerate", "safetensors"]
    )
    os.execv(sys.executable, [sys.executable, *sys.argv, "--deps-ready"])


# ---------------------------------------------------------------------------
# Pure NumPy FP64 / FP32 Reference Operators (aligned with smoke_checks.py)
# ---------------------------------------------------------------------------

import numpy as np


def sanitize_json(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(x) for x in obj]
    return obj


def np_norm(x) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def np_step(S: np.ndarray, k: np.ndarray, v: np.ndarray, beta, gamma) -> np.ndarray:
    decayed = S * gamma
    mem = np.sum(decayed * k[:, None], axis=0)
    delta = (v - mem) * beta
    return decayed + k[:, None] * delta[None, :]


def np_linear(S: np.ndarray, k: np.ndarray, beta, gamma) -> np.ndarray:
    decayed = gamma * S
    return decayed - (beta * k)[:, None] * (k @ decayed)[None, :]


def np_run(k_seq, v_seq, beta_seq, gamma_seq, initial=None, states: bool = False):
    dtype = k_seq.dtype
    d_k, d_v = k_seq.shape[1], v_seq.shape[1]
    S = np.zeros((d_k, d_v), dtype=dtype) if initial is None else initial.astype(dtype).copy()
    hist = [S.copy()] if states else None
    for k, v, b, g in zip(k_seq, v_seq, beta_seq, gamma_seq):
        S = np_step(S, k, v, b, g)
        if states:
            hist.append(S.copy())
    return (S, hist) if states else S


def np_unit_propagator(k_seq: np.ndarray, beta_seq: np.ndarray) -> np.ndarray:
    """Computes C = C_m ... C_1 without scalar gamma to avoid trivial underflow."""
    d_k = k_seq.shape[1]
    C = np.eye(d_k, dtype=np.float64)
    for k, b in zip(k_seq.astype(np.float64), beta_seq.astype(np.float64)):
        C = C - (b * k)[:, None] * (k @ C)[None, :]
    return C


def np_inverse_step(X: np.ndarray, k: np.ndarray, B: np.ndarray, beta, gamma) -> np.ndarray:
    dtype = X.dtype.type
    a2 = np.sum(k * k)
    alpha = dtype(beta / (dtype(1) - beta * a2))
    zeta = dtype(dtype(1) / gamma)
    Y = X - B
    c = np.sum(k[:, None] * Y, axis=0)
    H = (alpha * k)[:, None] * c[None, :]
    return zeta * (Y + H)


def np_silu(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    sig = np.where(x64 >= 0, 1.0 / (1.0 + np.exp(-x64)), np.exp(x64) / (1.0 + np.exp(x64)))
    return (x64 * sig).astype(x.dtype)


def np_l2(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + EPS) * scale


def np_conv4_from_torch_weight(raw_tc: np.ndarray, weight_c4: np.ndarray, bias_c: np.ndarray | None) -> np.ndarray:
    """Exact causal depthwise conv matching F.conv1d(..., padding=3)[:, :, :T] + SiLU.

    In PyTorch Conv1d with kernel_size=4 and padding=3 sliced to :T:
    out[t, c] = bias[c] + sum_{lag=0..3} weight[c, 3 - lag] * raw[t - lag, c]  (raw[<0] = 0).
    """
    out = np.zeros_like(raw_tc, dtype=np.float64)
    if bias_c is not None:
        out += bias_c.astype(np.float64)[None, :]
    w64 = weight_c4.astype(np.float64)
    r64 = raw_tc.astype(np.float64)
    T = r64.shape[0]
    for lag in range(4):
        out[lag:T] += r64[: T - lag] * w64[:, 3 - lag][None, :]
    return np_silu(out).astype(raw_tc.dtype)


class AffineTreeNP:
    def __init__(self, k_seq, v_seq, beta_seq, gamma_seq, C: int = 16):
        self.k = k_seq
        self.v = v_seq
        self.beta = beta_seq
        self.gamma = gamma_seq
        self.C = C
        self.T = len(k_seq)
        self.d_k = k_seq.shape[1]
        self.d_v = v_seq.shape[1]
        self.N = (self.T + C - 1) // C
        self.size = 1
        while self.size < self.N:
            self.size *= 2
        dtype = k_seq.dtype
        ident = (np.eye(self.d_k, dtype=dtype), np.zeros((self.d_k, self.d_v), dtype=dtype))
        self.nodes = [None] * (2 * self.size)
        for i in range(self.size):
            if i < self.N:
                lo, hi = i * C, min((i + 1) * C, self.T)
                P = np.eye(self.d_k, dtype=dtype)
                for kk, bb, gg in zip(k_seq[lo:hi], beta_seq[lo:hi], gamma_seq[lo:hi]):
                    P = np_linear(P, kk, bb, gg)
                H = np_run(k_seq[lo:hi], v_seq[lo:hi], beta_seq[lo:hi], gamma_seq[lo:hi])
                self.nodes[self.size + i] = (P, H)
            else:
                self.nodes[self.size + i] = ident
        for i in range(self.size - 1, 0, -1):
            Pa, Ha = self.nodes[2 * i]
            Pb, Hb = self.nodes[2 * i + 1]
            self.nodes[i] = (Pb @ Pa, Pb @ Ha + Hb)

    def _node_ids(self, l: int, r: int) -> list[int]:
        left, right = [], []
        l += self.size
        r += self.size
        while l < r:
            if l & 1:
                left.append(l)
                l += 1
            if r & 1:
                r -= 1
                right.append(r)
            l //= 2
            r //= 2
        return left + right[::-1]

    def apply(self, lo: int, hi: int, S: np.ndarray) -> tuple[np.ndarray, dict]:
        S = S.copy()
        steps = 0
        nodes = 0
        while lo < hi and lo % self.C:
            S = np_step(S, self.k[lo], self.v[lo], self.beta[lo], self.gamma[lo])
            lo += 1
            steps += 1
        stop = hi - hi % self.C
        if stop >= lo:
            for nid in self._node_ids(lo // self.C, stop // self.C):
                P, H = self.nodes[nid]
                S = P @ S + H
                nodes += 1
            lo = stop
        while lo < hi:
            S = np_step(S, self.k[lo], self.v[lo], self.beta[lo], self.gamma[lo])
            lo += 1
            steps += 1
        return S, {"token_steps": steps, "dense_nodes": nodes}


# ---------------------------------------------------------------------------
# Deterministic Prompt Builder with Strict Label & Filler Verification
# ---------------------------------------------------------------------------

SAFE_CODES = ["gamma", "delta", "omega", "sigma", "theta", "kappa", "lambda", "zeta"]


def find_subsequence(seq: list[int], sub: list[int]) -> int:
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if seq[i : i + m] == sub:
            return i
    return -1


def build_fixed_prompts(tokenizer) -> tuple[list[dict], dict]:
    ids_alpha = tokenizer.encode(" alpha", add_special_tokens=False)
    ids_beta = tokenizer.encode(" beta", add_special_tokens=False)
    if len(ids_alpha) != 1 or len(ids_beta) != 1 or ids_alpha[0] == ids_beta[0]:
        raise RuntimeError(f"LABEL_TOKENIZATION_MISMATCH: alpha={ids_alpha}, beta={ids_beta}")
    label_alpha_id = ids_alpha[0]
    label_beta_id = ids_beta[0]
    forbidden_ids = {
        label_alpha_id,
        label_beta_id,
        *tokenizer.encode("alpha", add_special_tokens=False),
        *tokenizer.encode("beta", add_special_tokens=False),
        *tokenizer.encode(" Alpha", add_special_tokens=False),
        *tokenizer.encode(" Beta", add_special_tokens=False),
    }

    rng = random.Random(SEED)
    prompts = []
    for task_id in range(6):
        nonce = f"R{task_id:02d}"
        is_json = task_id >= 3
        rule_text = f"Rule: If no surviving record for {nonce} exists, its code is beta. Answer with one word.\n"
        rule_ids = tokenizer.encode(rule_text, add_special_tokens=False)
        if label_beta_id not in rule_ids:
            raise RuntimeError("LABEL_TOKENIZATION_MISMATCH: label_beta_id missing from prefix rule")

        def make_filler_chunk(start_idx: int, count: int) -> str:
            parts = []
            for k in range(count):
                idx = start_idx + k
                tag = f"F{task_id:02d}_{idx:03d}"
                val = SAFE_CODES[(task_id * 7 + idx) % len(SAFE_CODES)]
                if is_json:
                    parts.append(f'{{"id": "{tag}", "code": "{val}"}}\n')
                else:
                    parts.append(f"Entry {tag}: the assigned word is {val}.\n")
            txt = "".join(parts)
            low = txt.lower()
            assert "alpha" not in low and "beta" not in low
            return txt

        filler_prefix_ids = tokenizer.encode(make_filler_chunk(1, 40), add_special_tokens=False)
        assert forbidden_ids.isdisjoint(filler_prefix_ids), "Forbidden label token leaked into prefix filler!"
        assert len(rule_ids) < 128
        prefix_ids = (rule_ids + filler_prefix_ids)[:128]
        assert len(prefix_ids) == 128

        needle_text = f"\nRecord {nonce}: the code is alpha.\n"
        needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
        if label_alpha_id not in needle_ids:
            raise RuntimeError("LABEL_TOKENIZATION_MISMATCH: label_alpha_id missing from needle_ids")

        span_pad_ids = tokenizer.encode(make_filler_chunk(100, 12), add_special_tokens=False)
        assert forbidden_ids.isdisjoint(span_pad_ids), "Forbidden label token leaked into span filler!"
        assert len(needle_ids) <= 32
        pad_needed = 32 - len(needle_ids)
        left_pad = pad_needed // 2
        right_pad = pad_needed - left_pad
        span_ids = span_pad_ids[:left_pad] + needle_ids + span_pad_ids[left_pad : left_pad + right_pad]
        assert len(span_ids) == 32
        assert find_subsequence(span_ids, needle_ids) >= 0

        filler_suffix_ids = tokenizer.encode(make_filler_chunk(200, 90), add_special_tokens=False)
        assert forbidden_ids.isdisjoint(filler_suffix_ids), "Forbidden label token leaked into suffix filler!"
        suffix_ids = filler_suffix_ids[:352]
        assert len(suffix_ids) == 352

        body_ids = prefix_ids + span_ids + suffix_ids
        assert len(body_ids) == BODY_LEN
        assert find_subsequence(body_ids[SPAN_START : SPAN_END + 1], needle_ids) >= 0

        query_text = f"\nQuestion: What is the code of {nonce}?\nAnswer:"
        query_ids = tokenizer.encode(query_text, add_special_tokens=False)
        assert 1 <= len(query_ids) <= 24

        detok_body = tokenizer.decode(body_ids, skip_special_tokens=False)
        reenc_body = tokenizer.encode(detok_body, add_special_tokens=False)

        prompts.append(
            {
                "task_id": task_id,
                "nonce": nonce,
                "style": "json" if is_json else "natural",
                "body_ids": body_ids,
                "query_ids": query_ids,
                "span_ inclusive": [SPAN_START, SPAN_END],
                "needle_text": needle_text,
                "query_text": query_text,
                "detok_preview_span": tokenizer.decode(span_ids),
                "reencode_len": len(reenc_body),
                "reencode_exact_match": reenc_body == body_ids,
                "body_sha256": hashlib.sha256(json.dumps(body_ids).encode()).hexdigest(),
            }
        )
        _ = rng.random()

    label_meta = {
        "label_alpha_str": " alpha",
        "label_beta_str": " beta",
        "label_alpha_id": label_alpha_id,
        "label_beta_id": label_beta_id,
    }
    return prompts, label_meta


# ---------------------------------------------------------------------------
# Main GPU Probe Execution
# ---------------------------------------------------------------------------


def run_probe(out_dir: Path, wall_budget_s: float = 1200.0) -> dict:
    t_wall_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qm

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available!")

    device = torch.device("cuda:0")
    use_bf16 = torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Startup check 1: Conv1d lag ordering against locked qm.causal_conv1d_fn
    with torch.inference_mode():
        w_test = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]], device=device, dtype=torch.float32)  # [1, 1, 4]
        x_test = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device, dtype=torch.float32)  # impulse at t=0
        y_test = qm.causal_conv1d_fn(x_test, w_test.squeeze(1), None, activation=None)
        expected = [40.0, 30.0, 20.0, 10.0]  # lag=0->w[3]=40, lag=1->w[2]=30, lag=2->w[1]=20, lag=3->w[0]=10
        if not np.allclose(y_test.cpu().numpy().ravel(), expected, atol=1e-6):
            raise RuntimeError(f"CONV_LAG_MISMATCH: got {y_test.cpu().numpy().ravel().tolist()}, expected {expected}")

    class TextProbe(torch.nn.Module):
        def __init__(self, full_model):
            super().__init__()
            self.model = full_model.model.language_model
            self.lm_head = full_model.lm_head
            self.config = full_model.config.text_config

        def forward(self, ids: torch.Tensor, cache=None, positions: torch.Tensor | None = None):
            out = self.model(
                input_ids=ids,
                past_key_values=cache,
                position_ids=positions,
                use_cache=True,
            )
            logits = self.lm_head(out.last_hidden_state[:, -1:]).float()
            return SimpleNamespace(
                logits=logits,
                last_hidden_last_token=out.last_hidden_state[:, -1:].detach().float().cpu(),
                past_key_values=out.past_key_values,
            )

    def verify_architecture(cfg) -> None:
        expected_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 6
        checks = [
            cfg.num_hidden_layers == 24,
            list(cfg.layer_types) == expected_types,
            cfg.linear_key_head_dim == 128,
            cfg.linear_value_head_dim == 128,
            cfg.linear_num_key_heads == 16,
            cfg.linear_num_value_heads == 16,
            cfg.linear_conv_kernel_dim == 4,
            cfg.num_attention_heads == 8,
            cfg.num_key_value_heads == 2,
            cfg.head_dim == 256,
            abs(float(cfg.rope_parameters.get("partial_rotary_factor", getattr(cfg, "partial_rotary_factor", 0.0))) - 0.25) < 1e-6,
        ]
        if not all(checks):
            raise RuntimeError(f"ARCH_MISMATCH: checks={checks}")

    def cache_digest(c: DynamicCache) -> str:
        h = hashlib.sha256()
        for l in FULL_ATTN_LAYERS:
            h.update(str(c.layers[l].get_seq_length()).encode())
            h.update(c.layers[l].keys.detach().float().cpu().numpy().tobytes()[:4096])
        for l in GDN_LAYERS:
            rs = c.get_recurrent_state(l, 0)
            cs = c.get_conv_state(l, 0)
            h.update(rs.detach().float().cpu().numpy().tobytes()[:4096])
            h.update(cs.detach().float().cpu().numpy().tobytes()[:1024])
        return h.hexdigest()

    def clone_cache(c: DynamicCache) -> DynamicCache:
        before = cache_digest(c)
        cloned = copy.deepcopy(c)
        after = cache_digest(c)
        if before != after or cache_digest(cloned) != before:
            raise RuntimeError("CACHE_CLONE_CORRUPTION")
        return cloned

    def evict_kv(c: DynamicCache, keep_idx: torch.Tensor) -> DynamicCache:
        for l in FULL_ATTN_LAYERS:
            row = c.layers[l]
            assert row.keys.shape[-2] == BODY_LEN
            ix = keep_idx.to(row.keys.device)
            row.keys = row.keys.index_select(-2, ix).contiguous()
            row.values = row.values.index_select(-2, ix).contiguous()
            assert row.get_seq_length() == len(keep_idx)
        return c

    def replace_gdn(c: DynamicCache, clean_c: DynamicCache) -> DynamicCache:
        for l in GDN_LAYERS:
            rs = clean_c.get_recurrent_state(l, 0)
            cs = clean_c.get_conv_state(l, 0)
            assert rs.dtype == torch.float32
            c.layers[l].recurrent_states[0].copy_(rs)
            c.layers[l].conv_states[0].copy_(cs)
        return c

    keep_np = np.r_[np.arange(0, SPAN_START), np.arange(SPAN_END + 1, BODY_LEN)]
    assert len(keep_np) == 480

    all_arms_rows = []
    model_summaries = {}

    pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()

    for tag, repo_id, revision in MODELS:
        if time.perf_counter() - t_wall_start > wall_budget_s - 30:
            model_summaries[tag] = {"status": "CENSORED_BUDGET_BEFORE_START"}
            continue

        mdir = out_dir / f"model_{tag}"
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "captured").mkdir(exist_ok=True)

        t_load0 = time.perf_counter()
        print(f"[{tag}] Loading {repo_id}@{revision[:10]} in {model_dtype}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
        prompts, label_meta = build_fixed_prompts(tokenizer)
        (mdir / "prompts.json").write_text(
            json.dumps({"model": repo_id, "revision": revision, "labels": label_meta, "prompts": prompts}, indent=2)
            + "\n"
        )

        full = Qwen3_5ForConditionalGeneration.from_pretrained(
            repo_id,
            revision=revision,
            dtype=model_dtype,
            attn_implementation="eager",
        )
        probe = TextProbe(full).eval().to(device)
        del full
        torch.cuda.empty_cache()
        t_load_s = time.perf_counter() - t_load0

        verify_architecture(probe.config)
        assert probe.lm_head.weight.device == device

        # Startup check 2: Verify RoPE uses explicit position_ids=keep rather than compact 0..479
        keep_t = torch.tensor(keep_np, dtype=torch.long, device=device).unsqueeze(0)
        compact_t = torch.arange(len(keep_np), dtype=torch.long, device=device).unsqueeze(0)
        sample_ids = torch.tensor([prompts[0]["body_ids"]], dtype=torch.long, device=device)[:, keep_t[0]]
        with torch.inference_mode():
            out_keep = probe(sample_ids, positions=keep_t)
            out_comp = probe(sample_ids, positions=compact_t)
            k_keep = out_keep.past_key_values.layers[3].keys
            k_comp = out_comp.past_key_values.layers[3].keys
            # Prefix 0..127 has identical positions -> identical Layer-3 keys!
            pref_diff = float((k_keep[:, :, :128, :64] - k_comp[:, :, :128, :64]).abs().max().item())
            # Suffix 128..479 has shifted positions (160..511 vs 128..479) -> rotated rotary dims MUST differ!
            suff_diff = float((k_keep[:, :, 128:, :64] - k_comp[:, :, 128:, :64]).abs().max().item())
            if pref_diff > 1e-5 or suff_diff < 1e-3:
                raise RuntimeError(f"POSITION_MASK_MISMATCH: pref_diff={pref_diff}, suff_diff={suff_diff}")
            del out_keep, out_comp

        # Setup deterministic recurrence & projection hooks
        orig_chunk = qm.torch_chunk_gated_delta_rule
        orig_recur = qm.torch_recurrent_gated_delta_rule
        active = {"layer": None, "enabled": False, "wrapper_calls": 0}
        captured: dict[int, dict] = {}
        handles = []

        def make_pre(l_idx: int):
            def hook(mod, args, kwargs):
                active["layer"] = l_idx
                if active["enabled"]:
                    x = args[0] if args else kwargs["hidden_states"]
                    captured.setdefault(l_idx, {})["input"] = x.detach().float().cpu()
            return hook

        def post_hook(mod, args, out):
            active["layer"] = None

        def make_proj(name: str):
            def hook(mod, args, out):
                l_idx = active["layer"]
                if active["enabled"] and l_idx is not None:
                    t = out.detach()
                    rec = captured.setdefault(l_idx, {})
                    rec[f"{name}_dtype"] = str(t.dtype)
                    if name == "qkv":
                        # Slice only CAPTURED_HEADS across Q (2048), K (2048), V (2048) to keep RAM small
                        q_raw, k_raw, v_raw = torch.split(t, [2048, 2048, 2048], dim=-1)
                        B_, L_, _ = q_raw.shape
                        rec["q_raw"] = q_raw.view(B_, L_, 16, 128)[:, :, CAPTURED_HEADS].float().cpu()
                        rec["k_raw"] = k_raw.view(B_, L_, 16, 128)[:, :, CAPTURED_HEADS].float().cpu()
                        rec["v_raw"] = v_raw.view(B_, L_, 16, 128)[:, :, CAPTURED_HEADS].float().cpu()
                    elif name == "z":
                        B_, L_, _ = t.shape
                        rec["z"] = t.view(B_, L_, 16, 128)[:, :, CAPTURED_HEADS].float().cpu()
                    elif name == "b":
                        # Capture sigmoid in native dtype AND fp32
                        rec["b_raw"] = t[:, :, CAPTURED_HEADS].float().cpu()
                        rec["beta_native"] = t[:, :, CAPTURED_HEADS].sigmoid().float().cpu()
                    elif name == "a":
                        rec["a_raw"] = t[:, :, CAPTURED_HEADS].float().cpu()
            return hook

        def wrap_kernel(fn):
            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                **kw,
            ):
                l_idx = active["layer"]
                if active["enabled"] and l_idx is not None:
                    active["wrapper_calls"] += 1
                    q, k = query.float(), key.float()
                    if use_qk_l2norm_in_kernel:
                        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
                        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
                    q = q / (q.shape[-1] ** 0.5)
                    rec = captured.setdefault(l_idx, {})
                    rec["q"] = q[:, :, CAPTURED_HEADS].detach().cpu()
                    rec["k"] = k[:, :, CAPTURED_HEADS].detach().cpu()
                    rec["v"] = value.float()[:, :, CAPTURED_HEADS].detach().cpu()
                    rec["g"] = g.float()[:, :, CAPTURED_HEADS].detach().cpu()
                    rec["beta"] = beta.float()[:, :, CAPTURED_HEADS].detach().cpu()
                    rec["beta_input_dtype"] = str(beta.dtype)
                ans = fn(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    **kw,
                )
                if active["enabled"] and l_idx is not None and ans[1] is not None:
                    captured[l_idx]["state_kernel"] = ans[1][:, CAPTURED_HEADS].detach().float().cpu()
                return ans
            return wrapped

        try:
            for l_idx in CAPTURED_LAYERS:
                attn_mod = probe.model.layers[l_idx].linear_attn
                handles.append(attn_mod.register_forward_pre_hook(make_pre(l_idx), with_kwargs=True))
                handles.append(attn_mod.register_forward_hook(post_hook))
                for pname in ["qkv", "a", "b", "z"]:
                    handles.append(getattr(attn_mod, f"in_proj_{pname}").register_forward_hook(make_proj(pname)))
            qm.torch_chunk_gated_delta_rule = wrap_kernel(orig_chunk)
            qm.torch_recurrent_gated_delta_rule = wrap_kernel(orig_recur)

            # Startup check 3: Hook invariance & exact wrapper call count
            test_in = torch.tensor([prompts[0]["body_ids"][:64]], dtype=torch.long, device=device)
            with torch.inference_mode():
                active["enabled"] = False
                ref_out = probe(test_in)
                captured.clear()
                active["wrapper_calls"] = 0
                active["enabled"] = True
                hook_out = probe(test_in)
                active["enabled"] = False
            if active["wrapper_calls"] != len(CAPTURED_LAYERS):
                raise RuntimeError(
                    f"HOOK_NOT_FIRED: wrapper_calls={active['wrapper_calls']} != {len(CAPTURED_LAYERS)}"
                )
            max_logit_diff = float((ref_out.logits - hook_out.logits).abs().max().item())
            max_state_diff = float(
                (
                    ref_out.past_key_values.get_recurrent_state(0, 0)
                    - hook_out.past_key_values.get_recurrent_state(0, 0)
                )
                .abs()
                .max()
                .item()
            )
            if max_logit_diff > 1e-5 or max_state_diff > 1e-6:
                raise RuntimeError(f"HOOK_PERTURBED_OUTPUT: logit_diff={max_logit_diff}, state_diff={max_state_diff}")
            captured.clear()
            del ref_out, hook_out

            meta = {
                "model_tag": tag,
                "repo_id": repo_id,
                "revision": revision,
                "transformers_version": transformers.__version__,
                "transformers_locked_commit": TRANSFORMERS_COMMIT,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "model_dtype": str(model_dtype),
                "use_hub_kernels_env": os.environ.get("USE_HUB_KERNELS"),
                "chunk_kernel_qualname": f"{orig_chunk.__module__}.{orig_chunk.__qualname__}",
                "recurrent_kernel_qualname": f"{orig_recur.__module__}.{orig_recur.__qualname__}",
                "hook_invariance_max_logit_diff": max_logit_diff,
                "rope_pref_diff": pref_diff,
                "rope_suff_diff": suff_diff,
                "load_seconds": round(t_load_s, 3),
                "seed": SEED,
                "pip_freeze_count": len(pip_freeze),
            }
            (mdir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

            mech_rows = []
            model_arm_rows = []
            alpha_id = label_meta["label_alpha_id"]
            beta_id = label_meta["label_beta_id"]

            # Pre-extract conv1d weights for CAPTURED_LAYERS & CAPTURED_HEADS
            conv_weights_by_layer = {}
            for l_idx in CAPTURED_LAYERS:
                w_full = probe.model.layers[l_idx].linear_attn.conv1d.weight.detach().float().cpu().numpy()[:, 0, :]
                # Split [6144, 4] -> q[2048, 4], k[2048, 4], v[2048, 4] -> select CAPTURED_HEADS [4, 128, 4]
                wq, wk, wv = np.split(w_full, [2048, 4096], axis=0)
                conv_weights_by_layer[l_idx] = {
                    "q": wq.reshape(16, 128, 4)[CAPTURED_HEADS],
                    "k": wk.reshape(16, 128, 4)[CAPTURED_HEADS],
                    "v": wv.reshape(16, 128, 4)[CAPTURED_HEADS],
                }

            for p_item in prompts:
                tid = p_item["task_id"]
                print(f"[{tag}] Running task {tid}/5...", flush=True)
                body_ids = torch.tensor([p_item["body_ids"]], dtype=torch.long, device=device)
                clean_ids = body_ids[:, keep_t[0]]
                query_ids = torch.tensor([p_item["query_ids"]], dtype=torch.long, device=device)
                q_len = query_ids.shape[1]
                pos_orig = torch.arange(BODY_LEN, dtype=torch.long, device=device).unsqueeze(0)
                pos_query_orig = torch.arange(BODY_LEN, BODY_LEN + q_len, dtype=torch.long, device=device).unsqueeze(0)
                pos_query_comp = torch.arange(480, 480 + q_len, dtype=torch.long, device=device).unsqueeze(0)

                with torch.inference_mode():
                    # 1) Prefill Original (O) with hooks enabled
                    captured.clear()
                    active["wrapper_calls"] = 0
                    active["enabled"] = True
                    pref_O = probe(body_ids, positions=pos_orig)
                    active["enabled"] = False
                    assert active["wrapper_calls"] == len(CAPTURED_LAYERS), "HOOK_NOT_FIRED on O"
                    cap_orig = copy.deepcopy(captured)
                    captured.clear()

                    # Verify recurrent cache dtype is FP32
                    assert pref_O.past_key_values.get_recurrent_state(0, 0).dtype == torch.float32

                    # 2) Prefill Clean Position-Preserved (C_pos) without hooks (for 6-arm cache ops)
                    pref_Cpos = probe(clean_ids, positions=keep_t)

                    # 3) Prefill Clean Compact (N_compact) with hooks enabled (for T5 cross-layer drift!)
                    active["wrapper_calls"] = 0
                    active["enabled"] = True
                    pref_Ncomp = probe(clean_ids, positions=compact_t)
                    active["enabled"] = False
                    assert active["wrapper_calls"] == len(CAPTURED_LAYERS), "HOOK_NOT_FIRED on N_compact"
                    cap_clean = copy.deepcopy(captured)
                    captured.clear()

                    # Evaluate 6 independent arms on identical query
                    arm_caches = {
                        "O": (clone_cache(pref_O.past_key_values), pos_query_orig),
                        "K": (evict_kv(clone_cache(pref_O.past_key_values), keep_t[0]), pos_query_orig),
                        "G": (replace_gdn(clone_cache(pref_O.past_key_values), pref_Cpos.past_key_values), pos_query_orig),
                        "B": (
                            replace_gdn(
                                evict_kv(clone_cache(pref_O.past_key_values), keep_t[0]),
                                pref_Cpos.past_key_values,
                            ),
                            pos_query_orig,
                        ),
                        "C_pos": (clone_cache(pref_Cpos.past_key_values), pos_query_orig),
                        "N_compact": (clone_cache(pref_Ncomp.past_key_values), pos_query_comp),
                    }
                    arm_results = {}
                    for arm_name, (acache, qpos) in arm_caches.items():
                        q_out = probe(query_ids, cache=acache, positions=qpos)
                        lvec = q_out.logits[0, 0]
                        la = float(lvec[alpha_id].item())
                        lb = float(lvec[beta_id].item())
                        top_id = int(torch.argmax(lvec).item())
                        arm_results[arm_name] = {
                            "logit_alpha": la,
                            "logit_beta": lb,
                            "margin": la - lb,
                            "argmax_id": top_id,
                            "argmax_str": tokenizer.decode([top_id]),
                            "hidden": q_out.last_hidden_last_token[0, 0].numpy(),
                        }

                    del pref_O, pref_Cpos, pref_Ncomp, arm_caches
                    torch.cuda.empty_cache()

                mO = arm_results["O"]["margin"]
                mK = arm_results["K"]["margin"]
                mG = arm_results["G"]["margin"]
                mB = arm_results["B"]["margin"]
                mC = arm_results["C_pos"]["margin"]
                mN = arm_results["N_compact"]["margin"]

                c_G = mK - mB
                c_F = mK - mO
                c_staleKV = mB - mC
                c_rope = mC - mN
                c_int = mK - mB - mO + mG
                ratio = abs(c_G) / abs(c_F) if abs(c_F) >= 1e-4 else None
                hidden_KB = np_norm(arm_results["K"]["hidden"] - arm_results["B"]["hidden"])
                hidden_BC = np_norm(arm_results["B"]["hidden"] - arm_results["C_pos"]["hidden"])

                arm_record = {
                    "model": tag,
                    "task_id": tid,
                    "style": p_item["style"],
                    "margins": {"O": mO, "K": mK, "G": mG, "B": mB, "C_pos": mC, "N_compact": mN},
                    "arms": {
                        k: {kk: vv for kk, vv in v.items() if kk != "hidden"} for k, v in arm_results.items()
                    },
                    "c_G": c_G,
                    "c_F": c_F,
                    "c_staleKV": c_staleKV,
                    "c_rope": c_rope,
                    "c_int": c_int,
                    "conflict_sign": bool(c_G * c_F < 0),
                    "GhostKV_logit_ratio": ratio,
                    "hidden_norm_K_minus_B": hidden_KB,
                    "hidden_norm_B_minus_Cpos": hidden_BC,
                }
                model_arm_rows.append(arm_record)
                all_arms_rows.append(arm_record)

                # Save compact NPZ snapshot for task 0
                if tid == 0:
                    for l_idx in CAPTURED_LAYERS:
                        np.savez_compressed(
                            mdir / "captured" / f"task{tid:02d}_l{l_idx:02d}.npz",
                            k=cap_orig[l_idx]["k"][0].numpy(),
                            v=cap_orig[l_idx]["v"][0].numpy(),
                            g=cap_orig[l_idx]["g"][0].numpy(),
                            beta=cap_orig[l_idx]["beta"][0].numpy(),
                            state_kernel=cap_orig[l_idx]["state_kernel"][0].numpy(),
                            beta_dtype=np.array(cap_orig[l_idx]["beta_input_dtype"]),
                        )

                # ---------------------------------------------------------------
                # P1, P2, P3, P4 Mechanistic Analysis on Captured Operands
                # ---------------------------------------------------------------
                for l_idx in CAPTURED_LAYERS:
                    co = cap_orig[l_idx]
                    cc = cap_clean[l_idx]
                    for h_pos, h_id in enumerate(CAPTURED_HEADS):
                        k32 = co["k"][0, :, h_pos].numpy().astype(np.float32)  # [512, 128]
                        v32 = co["v"][0, :, h_pos].numpy().astype(np.float32)
                        g32 = co["g"][0, :, h_pos].numpy().astype(np.float32)
                        beta32 = co["beta"][0, :, h_pos].numpy().astype(np.float32)
                        q32 = co["q"][0, :, h_pos].numpy().astype(np.float32)
                        gamma32 = np.exp(g32).astype(np.float32)

                        k64 = k32.astype(np.float64)
                        v64 = v32.astype(np.float64)
                        beta64 = beta32.astype(np.float64)
                        gamma64 = gamma32.astype(np.float64)

                        a2_all = np.sum(k64 * k64, axis=-1)
                        delta_all = 1.0 - beta64 * a2_all
                        beta_sat_count = int(np.sum(beta64 >= 1.0))
                        gamma_zero_count = int(np.sum(gamma64 <= 0.0))
                        outside_assump = int(np.sum((gamma64 <= 0.0) | (delta_all <= 0.0)))

                        S_full64, states64 = np_run(k64, v64, beta64, gamma64, states=True)
                        S_full32, states32 = np_run(k32, v32, beta32, gamma32, states=True)
                        S_kern32 = co["state_kernel"][0, h_pos].numpy().astype(np.float64)
                        kern_vs_oracle_rel = np_norm(S_kern32 - S_full64) / max(1.0, np_norm(S_full64))

                        scale_ST = max(1.0, np_norm(S_full64))
                        delta_inj = (2.0**-24) * scale_ST
                        tau_tol = 1e-2 * scale_ST

                        # --- P1: Spectral Survival-Recovery Duality & Recovery Horizon ---
                        p1_windows = []
                        for m in [1, 2, 4, 8, 16, 32, 64, 128]:
                            lo = BODY_LEN - m
                            ks, vs, bs, gs = k64[lo:], v64[lo:], beta64[lo:], gamma64[lo:]
                            ds = delta_all[lo:]
                            if np.any(gs <= 0.0) or np.any(ds <= 0.0):
                                p1_windows.append({"m": m, "status": "EFFECTIVE_OPERATOR_OUTSIDE_ASSUMPTIONS"})
                                continue
                            Lam = float(-np.sum(np.log(gs)))
                            E_sum = float(-np.sum(np.log(ds)))
                            r_alg_upper = min(m, 128)
                            k_sv = np.linalg.svd(ks, compute_uv=False)
                            r_num = int(np.sum(k_sv > 1e-10 * max(float(k_sv[0]), 1e-30)))

                            C_mat = np_unit_propagator(ks, bs)
                            U_c, sv_c, Vh_c = np.linalg.svd(C_mat)
                            cond_ok = bool(sv_c[-1] / max(sv_c[0], 1e-30) >= 1e-10)
                            chi_val = Lam - math.log(max(float(sv_c[-1]), 1e-300)) if cond_ok else None

                            # Worst-direction endpoint perturbation check on well-conditioned windows
                            inj_rel_err = None
                            if cond_ok and Lam < 600:
                                gamma_prod = math.exp(-Lam)
                                endpoint_N = np.zeros((128, 128), dtype=np.float64)
                                endpoint_N[:, 0] = delta_inj * U_c[:, -1]
                                rec_err = np_norm(np.linalg.solve(gamma_prod * C_mat, endpoint_N))
                                pred_rad = delta_inj * math.exp(chi_val)
                                inj_rel_err = abs(rec_err - pred_rad) / max(pred_rad, 1e-30)

                            # Natural FP32 rollback vs residual certificate Eq. (7) + 8-step checkpoint
                            shat = states32[BODY_LEN].copy()
                            B_sub32 = beta32[lo:, None, None] * k32[lo:, :, None] * v32[lo:, None, :]
                            back64 = {m: shat.astype(np.float64)}
                            valid_fp32_inv = True
                            fp32_inv_overflow_steps = None
                            with np.errstate(over="ignore", invalid="ignore"):
                                for j in range(m - 1, -1, -1):
                                    denom32 = np.float32(1.0) - beta32[lo + j] * np.sum(k32[lo + j] * k32[lo + j])
                                    if float(denom32) <= 0.0 or float(gamma32[lo + j]) <= 0.0:
                                        valid_fp32_inv = False
                                        break
                                    shat = np_inverse_step(shat, k32[lo + j], B_sub32[j], beta32[lo + j], gamma32[lo + j])
                                    if not np.all(np.isfinite(shat)):
                                        valid_fp32_inv = False
                                        fp32_inv_overflow_steps = m - j
                                        break
                                    back64[j] = shat.astype(np.float64)

                            nat_err = None
                            cert_slack = None
                            if valid_fp32_inv and cond_ok and (chi_val is not None and chi_val < 650):
                                nat_err = np_norm(back64[0] - states64[lo])
                                Z = np.eye(128, dtype=np.float64)
                                cert = 0.0
                                z_ok = True
                                with np.errstate(over="ignore", invalid="ignore"):
                                    for j in range(1, m + 1):
                                        kk, bb, gg = ks[j - 1], bs[j - 1], gs[j - 1]
                                        invA = (np.eye(128) + (bb / (1.0 - bb * float(kk @ kk))) * np.outer(kk, kk)) / gg
                                        Z = Z @ invA
                                        if not np.all(np.isfinite(Z)):
                                            z_ok = False
                                            break
                                        rho = np_step(back64[j - 1], kk, vs[j - 1], bb, gg) - back64[j]
                                        cert += float(np.linalg.norm(Z, ord=2)) * np_norm(rho)
                                if z_ok:
                                    cert += float(np.linalg.norm(Z, ord=2)) * np_norm(back64[m] - states64[BODY_LEN])
                                    if math.isfinite(cert):
                                        cert_slack = nat_err - cert

                            # 8-step checkpoint anchor replay error at endpoint
                            anc_idx = BODY_LEN - min(m, 8)
                            anc_end = np_run(
                                k32[anc_idx:], v32[anc_idx:], beta32[anc_idx:], gamma32[anc_idx:], initial=states32[anc_idx]
                            )
                            ckpt8_err = np_norm(anc_end.astype(np.float64) - states64[BODY_LEN])

                            p1_windows.append(
                                {
                                    "m": m,
                                    "status": "OK" if cond_ok else "CENSORED_CONDITIONING",
                                    "Lambda": Lam,
                                    "E": E_sum,
                                    "lower_bound_alg": Lam + E_sum / r_alg_upper,
                                    "upper_bound": Lam + E_sum,
                                    "r_num": r_num,
                                    "chi": chi_val,
                                    "pred_radius": (delta_inj * math.exp(chi_val)) if (cond_ok and chi_val is not None and chi_val < 650) else None,
                                    "above_tau": bool(chi_val >= math.log(tau_tol / delta_inj)) if (cond_ok and chi_val is not None) else True,
                                    "inj_rel_err": inj_rel_err,
                                    "natural_fp32_inv_err": nat_err,
                                    "natural_fp32_inv_overflow_step": fp32_inv_overflow_steps,
                                    "certificate_violation": cert_slack,
                                    "ckpt8_forward_err": ckpt8_err,
                                }
                            )

                        # --- P2: Uncensored Oblique Conjugation Theta via Forward Rank-1 Recurrence ---
                        p2_rows = []
                        for j_tok in [128, 144, 159]:
                            kj = k64[j_tok]
                            bj = float(beta64[j_tok])
                            gj = float(gamma64[j_tok])
                            vj = v64[j_tok]
                            aj2 = float(kj @ kj)
                            dj = 1.0 - bj * aj2
                            if gj <= 0.0 or dj <= 0.0:
                                continue
                            alpha_j = bj / dj
                            single_inv_gain = 1.0 / (gj * dj)
                            # Exact state difference at j+1 between including vs skipping token j
                            S_with_j = states64[j_tok + 1]
                            S_skip_j = states64[j_tok]
                            D_j = S_with_j - S_skip_j
                            norm_Dj = max(np_norm(D_j), 1e-30)

                            for m_suf in [4, 8, 16, 32, 64]:
                                hi_suf = j_tok + 1 + m_suf
                                ks_f = k64[j_tok + 1 : hi_suf]
                                bs_f = beta64[j_tok + 1 : hi_suf]
                                gs_f = gamma64[j_tok + 1 : hi_suf]
                                ds_f = delta_all[j_tok + 1 : hi_suf]
                                if np.any(gs_f <= 0.0) or np.any(ds_f <= 0.0):
                                    p2_rows.append(
                                        {"j": j_tok, "m": m_suf, "status": "EFFECTIVE_OPERATOR_OUTSIDE_ASSUMPTIONS"}
                                    )
                                    continue

                                # Chronological forward rank-1 biorthogonal recurrence (no matrix inverse!)
                                w_c = kj.copy()
                                z_c = kj.copy()
                                overflow = False
                                for kt, bt, dt in zip(ks_f, bs_f, ds_f):
                                    w_c = w_c - bt * float(kt @ w_c) * kt
                                    z_c = z_c + (bt / float(dt)) * float(kt @ z_c) * kt
                                    if not (np.all(np.isfinite(w_c)) and np.all(np.isfinite(z_c))):
                                        overflow = True
                                        break
                                if overflow:
                                    p2_rows.append({"j": j_tok, "m": m_suf, "status": "FLOAT64_OVERFLOW_EXTREME_THETA"})
                                    continue

                                biorth_err = abs(float(z_c @ w_c) - aj2)
                                J_val = np_norm(w_c) * np_norm(z_c)
                                # High-Theta safe closed form
                                if alpha_j * J_val > 1e6:
                                    theta_closed = (alpha_j * J_val) / gj
                                else:
                                    tt = 2.0 + 2.0 * alpha_j * aj2 + (alpha_j * J_val) ** 2
                                    disc = max(0.0, tt * tt - 4.0 / (dj * dj))
                                    theta_closed = math.sqrt((tt + math.sqrt(disc)) / 2.0) / gj

                                C_suf = np_unit_propagator(ks_f, bs_f)
                                sv_suf = np.linalg.svd(C_suf, compute_uv=False)
                                svd_well_cond = bool(sv_suf[-1] / max(sv_suf[0], 1e-30) >= 1e-10)
                                svd_rel_err = None
                                if svd_well_cond:
                                    M_mat = (np.eye(128) + alpha_j * np.outer(w_c, z_c)) / gj
                                    sv_M = np.linalg.svd(M_mat, compute_uv=False)
                                    svd_rel_err = abs(float(sv_M[0]) - theta_closed) / max(theta_closed, 1e-30)

                                Lam_suf = float(-np.sum(np.log(gs_f)))
                                gamma_suf = math.exp(-Lam_suf) if Lam_suf < 700 else 0.0
                                dir_ghost_ratio = np_norm(C_suf @ D_j) / norm_Dj
                                full_ghost_ratio = gamma_suf * dir_ghost_ratio
                                deleted_write_ctrl = bj * gamma_suf * np_norm(w_c) * np_norm(vj)

                                p2_rows.append(
                                    {
                                        "j": j_tok,
                                        "m": m_suf,
                                        "status": "OK",
                                        "svd_well_conditioned": svd_well_cond,
                                        "biorthogonal_err": biorth_err,
                                        "obliquity": J_val / max(aj2, 1e-30),
                                        "single_step_inv_gain": single_inv_gain,
                                        "theta_closed": theta_closed,
                                        "theta_over_single_step": theta_closed / single_inv_gain,
                                        "svd_vs_closed_rel_err": svd_rel_err,
                                        "directional_ghost_survival": dir_ghost_ratio,
                                        "full_ghost_survival": full_ghost_ratio,
                                        "deleted_write_norm": deleted_write_ctrl,
                                    }
                                )

                        # --- P3: Halo-Aware Single-Layer Scrub (Task 0..5 on Layer 0, plus Layers 0,1,2 Drift) ---
                        p3_info = {}
                        if tid == 0 or l_idx in [0, 1, 2]:
                            cw = conv_weights_by_layer[l_idx]
                            cw_k_h = cw["k"][h_pos]  # [128, 4]
                            cw_v_h = cw["v"][h_pos]  # [128, 4]
                            q_raw64 = co["q_raw"][0, :, h_pos].numpy().astype(np.float64)
                            k_raw64 = co["k_raw"][0, :, h_pos].numpy().astype(np.float64)
                            v_raw64 = co["v_raw"][0, :, h_pos].numpy().astype(np.float64)

                            # Exact FP64 F-counterfactual on fixed raw projections
                            k_orig_f64 = np_l2(np_conv4_from_torch_weight(k_raw64, cw_k_h, None))
                            v_orig_f64 = np_conv4_from_torch_weight(v_raw64, cw_v_h, None)
                            k_clean_f64 = np_l2(np_conv4_from_torch_weight(k_raw64[keep_np], cw_k_h, None))
                            v_clean_f64 = np_conv4_from_torch_weight(v_raw64[keep_np], cw_v_h, None)
                            b_clean_f64 = beta64[keep_np]
                            g_clean_f64 = gamma64[keep_np]

                            S_orig_f64, st_orig_f64 = np_run(k_orig_f64, v_orig_f64, beta64, gamma64, states=True)
                            S_clean_f64 = np_run(k_clean_f64, v_clean_f64, b_clean_f64, g_clean_f64)

                            h_end = min(BODY_LEN - 1, SPAN_END + 3)  # 162
                            halo_len = h_end - SPAN_END  # 3
                            S_prefix = st_orig_f64[SPAN_START]
                            S_clean_h = np_run(
                                k_clean_f64[SPAN_START : SPAN_START + halo_len],
                                v_clean_f64[SPAN_START : SPAN_START + halo_len],
                                b_clean_f64[SPAN_START : SPAN_START + halo_len],
                                g_clean_f64[SPAN_START : SPAN_START + halo_len],
                                initial=S_prefix,
                            )
                            D_h = st_orig_f64[h_end + 1] - S_clean_h
                            C_after_h = np_unit_propagator(k_orig_f64[h_end + 1 :], beta64[h_end + 1 :])
                            gam_after_h = float(np.prod(gamma64[h_end + 1 :]))
                            S_scrub_f64 = S_orig_f64 - gam_after_h * (C_after_h @ D_h)
                            fp64_scrub_rel_err = np_norm(S_scrub_f64 - S_clean_f64) / max(1.0, np_norm(S_clean_f64))

                            # Missing-halo negative control
                            S_no_halo = np_run(
                                k_orig_f64[SPAN_END + 1 :],
                                v_orig_f64[SPAN_END + 1 :],
                                beta64[SPAN_END + 1 :],
                                gamma64[SPAN_END + 1 :],
                                initial=S_prefix,
                            )
                            missing_halo_rel_err = np_norm(S_no_halo - S_clean_f64) / max(1.0, np_norm(S_clean_f64))

                            # FP32 AffineTree summary check
                            tree32 = AffineTreeNP(
                                k_orig_f64.astype(np.float32),
                                v_orig_f64.astype(np.float32),
                                beta32,
                                gamma32,
                                C=16,
                            )
                            p32_pre, _ = tree32.apply(0, SPAN_START, np.zeros((128, 128), dtype=np.float32))
                            p32_h = np_run(
                                k_clean_f64[SPAN_START : SPAN_START + halo_len].astype(np.float32),
                                v_clean_f64[SPAN_START : SPAN_START + halo_len].astype(np.float32),
                                b_clean_f64[SPAN_START : SPAN_START + halo_len].astype(np.float32),
                                g_clean_f64[SPAN_START : SPAN_START + halo_len].astype(np.float32),
                                initial=p32_pre,
                            )
                            p32_end, _ = tree32.apply(h_end + 1, BODY_LEN, p32_h)
                            fp32_tree_rel_err = np_norm(p32_end.astype(np.float64) - S_clean_f64) / max(
                                1.0, np_norm(S_clean_f64)
                            )

                            # Cross-layer F vs N_compact operand-defect bound Eq. (19) on retained 480 tokens
                            kn64 = cc["k"][0, :, h_pos].numpy().astype(np.float64)
                            vn64 = cc["v"][0, :, h_pos].numpy().astype(np.float64)
                            bn64 = cc["beta"][0, :, h_pos].numpy().astype(np.float64)
                            gn64 = np.exp(cc["g"][0, :, h_pos].numpy().astype(np.float64))
                            S_N_end, st_N = np_run(kn64, vn64, bn64, gn64, states=True)
                            _, st_F = np_run(k_clean_f64, v_clean_f64, b_clean_f64, g_clean_f64, states=True)

                            R_traj = max(np_norm(s) for s in st_N)
                            V_traj = max(
                                float(np.max(np.linalg.norm(v_clean_f64, axis=-1))),
                                float(np.max(np.linalg.norm(vn64, axis=-1))),
                            )
                            kappa_traj = max(
                                float(np.max(np.linalg.norm(k_clean_f64, axis=-1))),
                                float(np.max(np.linalg.norm(kn64, axis=-1))),
                            )
                            cert_e = 0.0
                            max_defect_violation = -math.inf
                            for t_step in range(480):
                                dk = np_norm(k_clean_f64[t_step] - kn64[t_step])
                                dv = np_norm(v_clean_f64[t_step] - vn64[t_step])
                                db = abs(float(b_clean_f64[t_step] - bn64[t_step]))
                                dg = abs(float(g_clean_f64[t_step] - gn64[t_step]))
                                a_step = dg + gn64[t_step] * ((kappa_traj**2) * db + 2.0 * kappa_traj * bn64[t_step] * dk)
                                b_step = db * kappa_traj * V_traj + bn64[t_step] * (V_traj * dk + kappa_traj * dv)
                                cert_e = g_clean_f64[t_step] * cert_e + R_traj * a_step + b_step
                                actual_e = np_norm(st_F[t_step + 1] - st_N[t_step + 1])
                                max_defect_violation = max(max_defect_violation, actual_e - cert_e)

                            p3_info = {
                                "fp64_scrub_rel_err": fp64_scrub_rel_err,
                                "fp32_tree_rel_err": fp32_tree_rel_err,
                                "missing_halo_rel_err": missing_halo_rel_err,
                                "F_vs_Ncompact_state_rel_diff": np_norm(S_clean_f64 - S_N_end)
                                / max(1.0, np_norm(S_N_end)),
                                "defect_bound_max_violation": max_defect_violation,
                                "defect_certificate_terminal": cert_e,
                            }

                            # Clipped halo control [508, 510] & timing on task0/l0/h0
                            if tid == 0 and l_idx == 0 and h_pos == 0:
                                keep_clip = np.r_[np.arange(0, 508), np.array([511])]
                                kc_clip = np_l2(np_conv4_from_torch_weight(k_raw64[keep_clip], cw_k_h, None))
                                vc_clip = np_conv4_from_torch_weight(v_raw64[keep_clip], cw_v_h, None)
                                S_clip_clean = np_run(kc_clip, vc_clip, beta64[keep_clip], gamma64[keep_clip])
                                S_clip_scrub = np_run(
                                    kc_clip[508:509],
                                    vc_clip[508:509],
                                    beta64[keep_clip][508:509],
                                    gamma64[keep_clip][508:509],
                                    initial=st_orig_f64[508],
                                )
                                p3_info["clipped_halo_508_510_rel_err"] = np_norm(S_clip_scrub - S_clip_clean) / max(
                                    1.0, np_norm(S_clip_clean)
                                )

                        # --- P4: Isotropic Spherical Key Control (task0 / l0 / h0 only) ---
                        p4_info = None
                        if tid == 0 and l_idx == 0 and h_pos == 0:
                            mc_rng = np.random.default_rng(SEED + 99)
                            m_mc = 32
                            trials_mc = 128
                            ks_mc = k64[BODY_LEN - m_mc :]
                            bs_mc = beta64[BODY_LEN - m_mc :]
                            gs_mc = gamma64[BODY_LEN - m_mc :]
                            rhos = np.linalg.norm(ks_mc, axis=-1)
                            b_eff = bs_mc * (rhos**2)
                            c_eff = b_eff * (2.0 - b_eff)
                            gam_prod2 = float(np.prod(gs_mc) ** 2)

                            # Unit rank-1 probe state from top right singular vector of real suffix
                            C_real32 = np_unit_propagator(ks_mc, bs_mc)
                            _, _, Vh_r = np.linalg.svd(C_real32)
                            x_init = Vh_r[0].copy()
                            f_vecs = np.repeat(x_init[None, :], trials_mc, axis=0)
                            inv_vecs = f_vecs.copy()
                            with np.errstate(over="ignore", invalid="ignore"):
                                for step_i in range(m_mc):
                                    u_rand = mc_rng.normal(size=(trials_mc, 128))
                                    u_rand /= np.linalg.norm(u_rand, axis=-1, keepdims=True)
                                    kk_r = u_rand * rhos[step_i]
                                    dot_f = np.sum(kk_r * f_vecs, axis=-1)
                                    f_vecs = (f_vecs - bs_mc[step_i] * kk_r * dot_f[:, None]) * gs_mc[step_i]
                                    if 1.0 - b_eff[step_i] > 0 and gs_mc[step_i] > 0:
                                        dot_i = np.sum(kk_r * inv_vecs, axis=-1)
                                        inv_vecs = (
                                            inv_vecs + (bs_mc[step_i] / (1.0 - b_eff[step_i])) * kk_r * dot_i[:, None]
                                        ) / gs_mc[step_i]
                                    else:
                                        inv_vecs.fill(np.nan)

                                f_en = np.sum(f_vecs * f_vecs, axis=-1)
                                i_en = np.sum(inv_vecs * inv_vecs, axis=-1)
                            pred_f = gam_prod2 * float(np.prod(1.0 - c_eff / 128.0))
                            inv_ok = bool(np.all(np.isfinite(i_en)) and gam_prod2 > 1e-300 and np.all(1.0 - b_eff > 0))
                            pred_i = (
                                (1.0 / gam_prod2) * float(np.prod(1.0 + ((1.0 - b_eff) ** -2 - 1.0) / 128.0))
                                if inv_ok
                                else None
                            )
                            real_f_en = float(np.sum((np.sqrt(gam_prod2) * (C_real32 @ x_init)) ** 2))
                            p4_info = {
                                "mc_ghost_mean": float(np.mean(f_en)),
                                "mc_ghost_se": float(np.std(f_en, ddof=1) / math.sqrt(trials_mc)),
                                "pred_ghost": pred_f,
                                "real_head_ghost_energy": real_f_en,
                                "mc_inv_status": "OK" if inv_ok else "MC_UNRESOLVED",
                                "mc_inv_mean": float(np.mean(i_en)) if inv_ok else None,
                                "mc_inv_se": float(np.std(i_en, ddof=1) / math.sqrt(trials_mc)) if inv_ok else None,
                                "pred_inv": pred_i,
                            }

                        mech_rows.append(
                            {
                                "model": tag,
                                "task_id": tid,
                                "layer": l_idx,
                                "head": h_id,
                                "beta_sat_count": beta_sat_count,
                                "gamma_zero_count": gamma_zero_count,
                                "outside_assumptions_count": outside_assump,
                                "kernel_vs_fp64_oracle_rel_err": kern_vs_oracle_rel,
                                "P1": p1_windows,
                                "P2": p2_rows,
                                "P3": p3_info,
                                "P4": p4_info,
                            }
                        )

                del cap_orig, cap_clean

            # Write JSONL logs for this model
            with (mdir / "arms.jsonl").open("w") as f_arms:
                for r in model_arm_rows:
                    f_arms.write(json.dumps(sanitize_json(r), allow_nan=False) + "\n")
            with (mdir / "mechanism.jsonl").open("w") as f_mech:
                for r in mech_rows:
                    f_mech.write(json.dumps(sanitize_json(r), allow_nan=False) + "\n")

            # Summarize per-model metrics
            p2_valid = [w for r in mech_rows for w in r["P2"] if w.get("status") == "OK"]
            p2_joint_hits = [
                w
                for w in p2_valid
                if w["theta_over_single_step"] >= 10.0 and w["directional_ghost_survival"] >= 0.25
            ]
            p2_full_hits = [
                w
                for w in p2_valid
                if w["theta_over_single_step"] >= 10.0 and w["full_ghost_survival"] >= 0.25
            ]
            max_biorth_err = max((w["biorthogonal_err"] for w in p2_valid), default=0.0)
            max_svd_err = max(
                (w["svd_vs_closed_rel_err"] for w in p2_valid if w["svd_vs_closed_rel_err"] is not None),
                default=0.0,
            )
            max_p3_fp64_err = max(
                (r["P3"]["fp64_scrub_rel_err"] for r in mech_rows if r["P3"]),
                default=0.0,
            )
            max_defect_viol = max(
                (r["P3"]["defect_bound_max_violation"] for r in mech_rows if r["P3"]),
                default=-math.inf,
            )
            conflicts_model = sum(
                1
                for r in model_arm_rows
                if abs(r["c_G"]) >= 1e-3 and abs(r["c_F"]) >= 1e-3 and (r["c_G"] * r["c_F"] < 0)
            )

            model_summaries[tag] = {
                "status": "COMPLETED",
                "tasks_run": len(model_arm_rows),
                "p2_valid_windows": len(p2_valid),
                "p2_joint_hit_fraction_directional": len(p2_joint_hits) / max(1, len(p2_valid)),
                "p2_joint_hit_fraction_with_scalar_decay": len(p2_full_hits) / max(1, len(p2_valid)),
                "p2_median_theta_over_single_step": float(
                    np.median([w["theta_over_single_step"] for w in p2_valid])
                )
                if p2_valid
                else None,
                "p2_p95_theta_over_single_step": float(
                    np.percentile([w["theta_over_single_step"] for w in p2_valid], 95)
                )
                if p2_valid
                else None,
                "p2_max_theta": max((w["theta_closed"] for w in p2_valid), default=None),
                "p2_max_biorthogonal_err": max_biorth_err,
                "p2_max_svd_vs_closed_rel_err": max_svd_err,
                "p3_max_fp64_scrub_rel_err": max_p3_fp64_err,
                "p3_max_defect_bound_violation": max_defect_viol,
                "p3_layer_drift_F_vs_Ncompact": {
                    f"layer_{l}": float(
                        np.mean(
                            [
                                r["P3"]["F_vs_Ncompact_state_rel_diff"]
                                for r in mech_rows
                                if r["layer"] == l and r["P3"]
                            ]
                        )
                    )
                    for l in [0, 1, 2]
                },
                "p5_conflict_count_of_6": conflicts_model,
                "mean_margins": {
                    arm_k: float(np.mean([r["margins"][arm_k] for r in model_arm_rows]))
                    for arm_k in ["O", "K", "G", "B", "C_pos", "N_compact"]
                },
                "mean_abs_effects": {
                    eff_k: float(np.mean([abs(r[eff_k]) for r in model_arm_rows]))
                    for eff_k in ["c_G", "c_F", "c_staleKV", "c_rope", "c_int"]
                },
            }

        finally:
            qm.torch_chunk_gated_delta_rule = orig_chunk
            qm.torch_recurrent_gated_delta_rule = orig_recur
            for h in handles:
                h.remove()
            active["enabled"] = False
            del probe
            torch.cuda.empty_cache()

    # Pooled P5 paired bootstrap (2000 resamples, seed-fixed)
    boot_rng = np.random.default_rng(SEED + 2027)
    signed_vals = np.array(
        [r["c_G"] * (1.0 if r["c_F"] >= 0 else -1.0) for r in all_arms_rows], dtype=np.float64
    )
    boot_means = []
    if len(signed_vals) > 0:
        for _ in range(2000):
            idx = boot_rng.integers(0, len(signed_vals), size=len(signed_vals))
            boot_means.append(float(np.mean(signed_vals[idx])))
    ci_low, ci_high = (
        (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
        if boot_means
        else (None, None)
    )
    pooled_conflicts = sum(
        1
        for r in all_arms_rows
        if abs(r["c_G"]) >= 1e-3 and abs(r["c_F"]) >= 1e-3 and (r["c_G"] * r["c_F"] < 0)
    )
    p5_gate_passed = bool(
        len(all_arms_rows) == 12
        and pooled_conflicts >= 10
        and ci_high is not None
        and ci_high < 0.0
        and all(model_summaries.get(m, {}).get("p5_conflict_count_of_6", 0) >= 4 for m, _, _ in MODELS)
    )

    summary = {
        "prereg_version": "R1.1",
        "seed": SEED,
        "total_wall_seconds": round(time.perf_counter() - t_wall_start, 3),
        "within_1200s_budget": (time.perf_counter() - t_wall_start) <= wall_budget_s,
        "models": model_summaries,
        "pooled_P5": {
            "n_samples": len(all_arms_rows),
            "conflict_count": pooled_conflicts,
            "signed_cG_sign_cF_mean": float(np.mean(signed_vals)) if len(signed_vals) else None,
            "bootstrap_95ci": [ci_low, ci_high],
            "preregistered_conflict_gate_passed": p5_gate_passed,
        },
    }
    summary = sanitize_json(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    # Create lightweight results archive (excluding bulky NPZ files for rapid download, plus full archive)
    tar_light = out_dir.parent / "probe_r1_results_light.tar.gz"
    with tarfile.open(tar_light, "w:gz") as tf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and p.suffix != ".npz":
                tf.add(p, arcname=p.relative_to(out_dir.parent))

    print("\n=== PROBE R1.1 SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/content/probe_r1_out")
    ap.add_argument("--budget-seconds", type=float, default=1200.0)
    ap.add_argument("--deps-ready", action="store_true")
    ap.add_argument("--syntax-check-only", action="store_true")
    args = ap.parse_args()

    if args.syntax_check_only:
        print("SYNTAX_AND_NUMPY_IMPORT_OK")
        return

    ensure_transformers_installed(allow_install=not args.deps_ready)
    run_probe(Path(args.out_dir), wall_budget_s=args.budget_seconds)


if __name__ == "__main__":
    main()
