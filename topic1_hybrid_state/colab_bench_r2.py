#!/usr/bin/env python3
"""Round 2.1 CHymERA benchmark. See the frozen R2.1 plan and theory.

System methods use N_gapped as primary oracle; mechanistic F ablations are
reported separately. CPU self-tests never download weights or write results
outside a temporary directory. GPU runs require the locked eager implementation.
"""

from __future__ import annotations

import os

os.environ["USE_HUB_KERNELS"] = "OFF"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import random
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TRANSFORMERS_COMMIT = "0f7a8768733c448751f25abcb4b6b3ffef829cf5"
TRANSFORMERS_TARBALL = f"https://github.com/huggingface/transformers/archive/{TRANSFORMERS_COMMIT}.tar.gz"
LOCKED_MODELING_SHA256 = "4463b8a108f678c166755101d95f188179f54143305953d78147de0389b80ebc"

MODEL_REGISTRY = {
    "0.8B": {
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "is_instruct": True,
        "expected": {"layers": 24, "h_k": 16, "h_v": 16, "h_kv": 2, "d_model": 1024},
    },
    "2B": {
        "repo_id": "Qwen/Qwen3.5-2B",
        "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "is_instruct": True,
        "expected": {"layers": 24, "h_k": 16, "h_v": 16, "h_kv": 2, "d_model": 2048},
    },
    "4B": {
        "repo_id": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "is_instruct": True,
        "expected": {"layers": 32, "h_k": 16, "h_v": 32, "h_kv": 4, "d_model": 2560},
    },
    "9B": {
        "repo_id": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "is_instruct": True,
        "expected": {"layers": 32, "h_k": 16, "h_v": 32, "h_kv": 4, "d_model": 4096},
    },
    "2B-Base": {
        "repo_id": "Qwen/Qwen3.5-2B-Base",
        "revision": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
        "is_instruct": False,
        "expected": {"layers": 24, "h_k": 16, "h_v": 16, "h_kv": 2, "d_model": 2048},
    },
}

SEED = 20261109
CONV_W = 4
EPS = 1e-6
PREREG_TAU_GATE = 0.05
TAU_SWEEP = [float("inf"), 0.20, 0.10, 0.05, 0.025, 0.01, 0.0]
DEFAULT_WALL_BUDGET_S = 28800.0
PREREG_REQUIRED_FILES = (
    "METHOD_AND_THEORY_R2.md",
    "COLAB_BENCH_R2_PLAN.md",
    "smoke_r2.py",
    "SMOKE_R2_RESULTS.json",
    "colab_bench_r2.py",
)


def verify_prereg_manifest(manifest_path: str | Path, required_files: tuple[str, ...] = PREREG_REQUIRED_FILES) -> dict:
    """Verify SHA-256 digests of all frozen preregistered files; hard-abort immediately on mismatch."""
    mpath = Path(manifest_path).resolve()
    if not mpath.is_file():
        raise RuntimeError(f"PREREG_MANIFEST_MISSING: {mpath}")
    base_dir = mpath.parent
    entries: dict[str, str] = {}
    for raw_line in mpath.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise RuntimeError(f"PREREG_MANIFEST_MALFORMED_LINE: {raw_line!r}")
        digest, rel_name = parts[0].strip().lower(), parts[1].strip().lstrip("*").strip()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise RuntimeError(f"PREREG_MANIFEST_INVALID_SHA256: {raw_line!r}")
        if rel_name in entries:
            raise RuntimeError(f"PREREG_MANIFEST_DUPLICATE_ENTRY: {rel_name}")
        entries[rel_name] = digest
    if set(entries.keys()) != set(required_files):
        raise RuntimeError(
            f"PREREG_MANIFEST_FILE_SET_MISMATCH: got {sorted(entries.keys())}, expected {sorted(required_files)}"
        )
    verified: dict[str, str] = {}
    for rel_name in required_files:
        fpath = base_dir / rel_name
        if not fpath.is_file():
            raise RuntimeError(f"PREREG_FILE_MISSING: {fpath}")
        actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
        expected = entries[rel_name]
        if actual != expected:
            raise RuntimeError(
                f"PREREG_SHA256_MISMATCH for {rel_name}: expected {expected}, got {actual}"
            )
        verified[rel_name] = actual
    print(f"[prereg] Verified SHA-256 manifest {mpath} ({len(verified)} files): PASS", flush=True)
    return {"manifest_path": str(mpath), "verified_sha256": verified}


# ---------------------------------------------------------------------------
# Pure NumPy / Algorithmic Core: Theorem 6, Theorem 7, Theorem 8
# ---------------------------------------------------------------------------


def sanitize_json(obj):
    if hasattr(obj, "detach"):
        return sanitize_json(obj.detach().cpu().tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else str(val)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(x) for x in obj]
    return obj


def canonicalize_spans(spans: list[tuple[int, int]], T: int) -> list[tuple[int, int]]:
    """Merges overlapping or adjacent inclusive spans [s, e] into canonical disjoint spans."""
    if not spans:
        return []
    cleaned = []
    for s, e in spans:
        if not (0 <= s <= e < T):
            raise ValueError(f"Invalid span [{s}, {e}] for T={T}")
        cleaned.append((int(s), int(e)))
    cleaned.sort(key=lambda x: (x[0], x[1]))
    merged = [cleaned[0]]
    for s, e in cleaned[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def build_survivor_geometry(spans: list[tuple[int, int]], T: int, W: int = CONV_W) -> dict:
    """Implements Theorem 7's exact survivor index map pi and coalesced seam halo H_W(D)."""
    canon = canonicalize_spans(spans, T)
    deleted_mask = np.zeros(T, dtype=bool)
    for s, e in canon:
        deleted_mask[s : e + 1] = True
    pi = np.flatnonzero(~deleted_mask).astype(np.int64)
    Lp = int(len(pi))

    def pi_ext(idx: int) -> int:
        return int(pi[idx]) if idx >= 0 else int(idx)

    halo_def = [i for i in range(Lp) if pi_ext(i) - pi_ext(i - (W - 1)) > (W - 1)]

    k_r_list = []
    halo_union_set = set()
    for s_r, _e_r in canon:
        k_r = int(np.sum(~deleted_mask[:s_r]))
        k_r_list.append(k_r)
        for u in range(k_r, min(Lp, k_r + W - 1)):
            halo_union_set.add(u)
    halo_union = sorted(halo_union_set)
    if halo_def != halo_union:
        raise RuntimeError(f"THEOREM7_SET_MISMATCH: def={halo_def} vs union={halo_union}")
    if len(halo_def) > (W - 1) * len(canon):
        raise RuntimeError("THEOREM7_CARDINALITY_VIOLATION")

    # Verify Smallest-Broken-Lag property: t* = pi(i) - m_min is ALWAYS in D
    for i in halo_def:
        broken = [m for m in range(1, W) if pi_ext(i - m) != pi_ext(i) - m]
        if not broken:
            raise RuntimeError(f"Halo index {i} has no broken lag")
        m_min = min(broken)
        t_star = pi_ext(i) - m_min
        if not (0 <= t_star < T and deleted_mask[t_star]):
            raise RuntimeError(f"THEOREM7_BROKEN_LAG_VIOLATION: i={i}, m_min={m_min}, t*={t_star}")

    island_lengths = [canon[r + 1][0] - canon[r][1] - 1 for r in range(len(canon) - 1)]
    return {
        "canonical_spans": canon,
        "deleted_mask": deleted_mask,
        "pi": pi,
        "L_prime": Lp,
        "k_r": k_r_list,
        "halo_indices": halo_def,
        "island_lengths": island_lengths,
    }


def partition_macro_chunks(T: int, C: int, W: int = CONV_W) -> list[tuple[int, int]]:
    """Partition [0, T) into chunks of at most C (including short tails)."""
    if C < W or T < W:
        raise ValueError(f"Require C >= {W} and T >= {W}, got C={C}, T={T}")
    chunks = []
    pos = 0
    while pos < T:
        nxt = min(T, pos + C)
        chunks.append((pos, nxt))
        pos = nxt
    return chunks


def plan_chymera_execution(T: int, C: int, spans: list[tuple[int, int]], W: int = CONV_W) -> dict:
    """Implements Theorem 6(b) exact dirty/clean/dropped chunk classification and execution segments."""
    geom = build_survivor_geometry(spans, T, W=W)
    deleted_mask = geom["deleted_mask"]
    chunks = partition_macro_chunks(T, C, W=W)

    # Map original index -> survivor index (-1 if deleted)
    orig_to_surv = np.full(T, -1, dtype=np.int64)
    orig_to_surv[~deleted_mask] = np.arange(geom["L_prime"], dtype=np.int64)

    chunk_records = []
    for c_idx, (lo, hi) in enumerate(chunks):
        if bool(np.all(deleted_mask[lo:hi])):
            status = "dropped"
            u_lo, u_hi = -1, -1
        else:
            ext_lo = max(0, lo - (W - 1))
            is_dirty = bool(np.any(deleted_mask[ext_lo:hi]))
            status = "dirty" if is_dirty else "clean"
            surv_in_chunk = orig_to_surv[lo:hi]
            surv_valid = surv_in_chunk[surv_in_chunk >= 0]
            u_lo = int(surv_valid[0])
            u_hi = int(surv_valid[-1]) + 1
            if u_hi - u_lo != len(surv_valid):
                raise RuntimeError("Non-contiguous survivor mapping inside chunk")
        chunk_records.append(
            {"c": c_idx, "orig_lo": lo, "orig_hi": hi, "status": status, "surv_lo": u_lo, "surv_hi": u_hi}
        )

    # Verify every halo survivor index belongs to a dirty chunk
    dirty_surv_set = set()
    for rec in chunk_records:
        if rec["status"] == "dirty":
            dirty_surv_set.update(range(rec["surv_lo"], rec["surv_hi"]))
    for h_i in geom["halo_indices"]:
        if h_i not in dirty_surv_set:
            raise RuntimeError(f"THEOREM6B_HALO_LEAK: halo survivor {h_i} landed in a clean chunk!")

    # Build execution schedule starting after maximal clean prefix
    first_non_clean = 0
    while first_non_clean < len(chunk_records) and chunk_records[first_non_clean]["status"] == "clean":
        first_non_clean += 1

    prefix_surv_len = (
        0 if first_non_clean == 0 else chunk_records[first_non_clean - 1]["surv_hi"]
    )
    segments = []
    idx = first_non_clean
    while idx < len(chunk_records):
        st = chunk_records[idx]["status"]
        if st == "dropped":
            idx += 1
            continue
        if st == "clean":
            c_list = []
            u_lo = chunk_records[idx]["surv_lo"]
            u_hi = u_lo
            while idx < len(chunk_records) and chunk_records[idx]["status"] == "clean":
                c_list.append(chunk_records[idx]["c"])
                u_hi = chunk_records[idx]["surv_hi"]
                idx += 1
            segments.append({"kind": "clean_bmm", "chunks": c_list, "surv_lo": u_lo, "surv_hi": u_hi})
        else:
            # Coalesce consecutive dirty chunks across any intervening dropped chunks!
            start_c = chunk_records[idx]["c"]
            u_lo = chunk_records[idx]["surv_lo"]
            u_hi = chunk_records[idx]["surv_hi"]
            c_list = [start_c]
            idx += 1
            while idx < len(chunk_records) and chunk_records[idx]["status"] in ("dirty", "dropped"):
                if chunk_records[idx]["status"] == "dirty":
                    c_list.append(chunk_records[idx]["c"])
                    u_hi = chunk_records[idx]["surv_hi"]
                idx += 1
            segments.append(
                {"kind": "dirty_replay", "anchor_chunk": start_c, "chunks": c_list, "surv_lo": u_lo, "surv_hi": u_hi}
            )

    replayed_tokens = sum(seg["surv_hi"] - seg["surv_lo"] for seg in segments if seg["kind"] == "dirty_replay")
    bound = min(geom["L_prime"], len(geom["canonical_spans"]) * (2 * C + W - 3))
    if replayed_tokens > bound:
        raise AssertionError(f"PROP6B_BOUND_VIOLATION: {replayed_tokens} > {bound}")
    clean_bmm_chunks = sum(len(seg["chunks"]) for seg in segments if seg["kind"] == "clean_bmm")
    return {
        "geometry": geom,
        "chunks": chunk_records,
        "first_non_clean_chunk": first_non_clean,
        "prefix_surv_len": prefix_surv_len,
        "segments": segments,
        "replayed_tokens": replayed_tokens,
        "clean_bmm_chunks": clean_bmm_chunks,
    }


# ---------------------------------------------------------------------------
# Pure NumPy Reference Operators for CPU Self-Test & Mechanistic Diagnostics
# ---------------------------------------------------------------------------


def np_wy_chunk_summary(
    k_seq: np.ndarray,
    v_seq: np.ndarray,
    g_seq: np.ndarray,
    beta_seq: np.ndarray,
    sub_chunk: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy FP64 reference of Theorem 6(a) UT-WY chunk folding for [H_v, L_c, d]."""
    H_v, L_c, d_k = k_seq.shape
    d_v = v_seq.shape[-1]
    pad = (sub_chunk - (L_c % sub_chunk)) % sub_chunk
    if pad > 0:
        k_seq = np.pad(k_seq, ((0, 0), (0, pad), (0, 0)))
        v_seq = np.pad(v_seq, ((0, 0), (0, pad), (0, 0)))
        g_seq = np.pad(g_seq, ((0, 0), (0, pad)))
        beta_seq = np.pad(beta_seq, ((0, 0), (0, pad)))
    n_sub = k_seq.shape[1] // sub_chunk
    k_b = k_seq.reshape(H_v, n_sub, sub_chunk, d_k)
    v_b = v_seq.reshape(H_v, n_sub, sub_chunk, d_v)
    g_b = g_seq.reshape(H_v, n_sub, sub_chunk)
    b_b = beta_seq.reshape(H_v, n_sub, sub_chunk)

    P_c = np.broadcast_to(np.eye(d_k, dtype=np.float64), (H_v, d_k, d_k)).copy()
    H_c = np.zeros((H_v, d_k, d_v), dtype=np.float64)
    tril_strict = np.tril(np.ones((sub_chunk, sub_chunk), dtype=bool), k=-1)

    for s in range(n_sub):
        ks = k_b[:, s]
        vs = v_b[:, s]
        gs = g_b[:, s]
        bs = b_b[:, s]
        cum_g = np.cumsum(gs, axis=-1)
        diff = cum_g[:, :, None] - cum_g[:, None, :]
        pw = np.where(tril_strict[None, :, :], np.exp(np.minimum(diff, 0.0)), 0.0)
        kb = ks * bs[:, :, None]
        vb = vs * bs[:, :, None]
        M = np.broadcast_to(np.eye(sub_chunk, dtype=np.float64), (H_v, sub_chunk, sub_chunk)).copy()
        M += np.matmul(kb, np.swapaxes(ks, -1, -2)) * pw
        dec_kb = kb * np.exp(cum_g)[:, :, None]
        u_sub = np.linalg.solve(M, vb)
        y_sub = np.linalg.solve(M, dec_kb)
        k_end = ks * np.exp(cum_g[:, -1:] - cum_g)[:, :, None]
        gam = np.exp(cum_g[:, -1])

        P_c = gam[:, None, None] * P_c - np.matmul(np.swapaxes(k_end, -1, -2), np.matmul(y_sub, P_c))
        H_c = gam[:, None, None] * H_c + np.matmul(
            np.swapaxes(k_end, -1, -2), u_sub - np.matmul(y_sub, H_c)
        )
    return P_c, H_c


def np_adjoint_readout_check(
    k1: np.ndarray,
    v1: np.ndarray,
    g1: np.ndarray,
    b1: np.ndarray,
    k2: np.ndarray,
    v2: np.ndarray,
    g2: np.ndarray,
    b2: np.ndarray,
    q_T: np.ndarray,
    S0_1: np.ndarray,
    S0_2: np.ndarray,
) -> dict:
    """Verifies Theorem 8 exact backward vector adjoint identity & tightness over Frobenius."""
    m, d_k = k1.shape
    S1 = S0_1.astype(np.float64).copy()
    S2 = S0_2.astype(np.float64).copy()
    S1_hist = [S1.copy()]
    defects = []
    gam2 = np.exp(g2.astype(np.float64))
    gam1 = np.exp(g1.astype(np.float64))

    for t in range(m):
        S1_prev = S1.copy()
        S1 = gam1[t] * (S1 - b1[t] * np.outer(k1[t], k1[t] @ S1)) + b1[t] * np.outer(k1[t], v1[t])
        S2 = gam2[t] * (S2 - b2[t] * np.outer(k2[t], k2[t] @ S2)) + b2[t] * np.outer(k2[t], v2[t])
        S2_on_S1 = gam2[t] * (S1_prev - b2[t] * np.outer(k2[t], k2[t] @ S1_prev)) + b2[t] * np.outer(k2[t], v2[t])
        defects.append(S1 - S2_on_S1)
        S1_hist.append(S1.copy())

    p = q_T.astype(np.float64).copy()
    rhs = np.zeros(S1.shape[1], dtype=np.float64)
    b_adj_cauchy = 0.0
    b_adj_dir = 0.0
    b_frob = 0.0
    cum_gamma = 1.0
    q_norm = float(np.linalg.norm(q_T))

    for t in range(m - 1, -1, -1):
        d_loc = defects[t]
        d_norm = float(np.linalg.norm(d_loc))
        proj_vec = p @ d_loc
        rhs += proj_vec
        b_adj_dir += float(np.linalg.norm(proj_vec))
        b_adj_cauchy += float(np.linalg.norm(p)) * d_norm
        b_frob += q_norm * cum_gamma * d_norm
        p = gam2[t] * (p - b2[t] * float(k2[t] @ p) * k2[t])
        cum_gamma *= float(gam2[t])

    dS0 = S0_1 - S0_2
    dS0_norm = float(np.linalg.norm(dS0))
    rhs += p @ dS0
    b_adj_dir += float(np.linalg.norm(p @ dS0))
    b_adj_cauchy += float(np.linalg.norm(p)) * dS0_norm
    b_frob += q_norm * cum_gamma * dS0_norm

    lhs = q_T.astype(np.float64) @ (S1 - S2)
    actual = float(np.linalg.norm(lhs))
    ident_err = float(np.linalg.norm(lhs - rhs)) / max(1.0, actual, b_adj_cauchy)
    return {
        "actual_readout_err": actual,
        "identity_rel_err": ident_err,
        "B_adj_dir": b_adj_dir,
        "B_adj_cauchy": b_adj_cauchy,
        "B_frob": b_frob,
        "violation_dir": actual - (b_adj_dir + 1e-12),
        "violation_cauchy": actual - (b_adj_cauchy + 1e-12),
        "tightness_dir_vs_frob": b_frob / max(b_adj_dir, 1e-30),
        "tightness_cauchy_vs_frob": b_frob / max(b_adj_cauchy, 1e-30),
    }


# ---------------------------------------------------------------------------
# Deterministic 72-Instance Benchmark Dataset Builder (Tasks A, B, C)
# ---------------------------------------------------------------------------

COLOR_WORDS = [
    "amber", "cobalt", "emerald", "ivory", "jasper", "linen",
    "ochre", "pearl", "quartz", "sable", "topaz", "velvet",
]
ROLE_WORDS = [
    "anchor", "beacon", "cipher", "daemon", "engine", "fabric",
    "gateway", "harbor", "kernel", "ledger", "matrix", "nexus",
]
SAFE_FILLER_TOPICS = [
    "Telemetry heartbeat nominal across worker pool; packet jitter remained within twelve microseconds.",
    "Scheduler checkpoint completed rotating daily log partitions onto cold object storage.",
    "Cache warmup verified deterministic checksum parity across regional mirror replicas.",
    "Ingress load balancer drained idle keepalive sockets without dropping active RPC streams.",
    "Compiler pass finished loop-invariant code motion on auxiliary tensor serialization routines.",
    "Audit daemon confirmed read-only mount permissions on container rootfs snapshot.",
    "Metrics collector flushed histogram buckets to local time-series ring buffer.",
    "Replication monitor observed zero quorum retries during routine leader lease renewal.",
]


class FastTokenizerAdapter:
    """Unifies HuggingFace `PreTrainedTokenizerBase`, `tokenizers.Tokenizer`, and mock tokenizers."""

    def __init__(self, tok):
        self._tok = tok

    def encode_plain(self, text: str) -> list[int]:
        res = self._tok.encode(text, add_special_tokens=False)
        return list(res.ids) if hasattr(res, "ids") else list(res)

    def decode_plain(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


def _make_identifier(idx: int, variant: int) -> str:
    c = COLOR_WORDS[(idx * 3 + variant * 5) % len(COLOR_WORDS)]
    r = ROLE_WORDS[(idx * 5 + variant * 7 + 1) % len(ROLE_WORDS)]
    num = 1100 + ((idx * 137 + variant * 521) % 8800)
    # Avoid any digit overlap with filler (filler uses words only, zero digits!)
    return f"{c}-{num}-{r}"


def _build_filler_lines(tok: FastTokenizerAdapter, needed_tokens: int, seed_offset: int, forbidden_strs: list[str]) -> list[int]:
    """Builds newline-terminated filler token sequences of exact length `needed_tokens` with zero forbidden leakage."""
    if needed_tokens <= 0:
        return []
    lines = []
    cur_ids: list[int] = []
    step = 0
    while len(cur_ids) < needed_tokens + 64:
        sent = SAFE_FILLER_TOPICS[(seed_offset + step * 3) % len(SAFE_FILLER_TOPICS)]
        line = f"Log note: {sent}\n"
        low = line.lower()
        for f in forbidden_strs:
            if f.lower() in low:
                raise RuntimeError(f"FILLER_LEAKAGE_ERROR: {f} in {line}")
        cur_ids.extend(tok.encode_plain(line))
        lines.append(line)
        step += 1

    nl_ids = tok.encode_plain("\n")
    assert len(nl_ids) == 1, f"Expected single-token newline, got {nl_ids}"
    nl_id = nl_ids[0]
    pad_word_ids = tok.encode_plain(" status")
    assert len(pad_word_ids) == 1

    # Trim cleanly so the last token is ALWAYS `\n` (`nl_id`) without creating `\n\n` (`[nl_id, nl_id]`)!
    if needed_tokens == 1:
        return [pad_word_ids[0]]
    trimmed = list(cur_ids[: needed_tokens - 1])
    if trimmed[-1] == nl_id:
        trimmed[-1] = pad_word_ids[0]
    candidate = trimmed + [nl_id]
    detok = tok.decode_plain(candidate)
    for f in forbidden_strs:
        if f.lower() in detok.lower():
            raise RuntimeError(f"FILLER_LEAKAGE_ERROR after slice: {f}")
    assert len(candidate) == needed_tokens and candidate[-1] == nl_id
    return candidate


def build_benchmark_dataset(raw_tokenizer, is_instruct: bool = True, max_instances: int | None = None) -> list[dict]:
    """Constructs all N=72 pre-registered instances across Tasks A, B, C and T in {1024, 2048, 4096}."""
    tok = FastTokenizerAdapter(raw_tokenizer)
    nl_id = tok.encode_plain("\n")[0]
    # In Qwen3.5 BPE, "---" is 1 token ([4277]) and "---\n" is 2 tokens ([4277, 198]);
    # neither merges across adjacent newline-terminated or bracket/letter-initial spans!
    island1_ids = tok.encode_plain("---")
    island2_ids = tok.encode_plain("---\n")
    if hasattr(raw_tokenizer, "to_str") or hasattr(raw_tokenizer, "vocab_size"):
        assert island1_ids == [4277] and len(island1_ids) == 1, "DELIMITER_TOKEN_MISMATCH"
        assert island2_ids == [4277, 198] and len(island2_ids) == 2, "DELIMITER_TOKEN_MISMATCH"
    assert len(island1_ids) == 1 and len(island2_ids) == 2

    dataset: list[dict] = []
    lengths = [1024, 2048, 4096] if is_instruct else [1024, 2048]
    tasks = ["TaskA_FactRetract", "TaskB_RAGEvict", "TaskC_SlidingScrub"]
    inst_global = 0

    for t_idx, task_name in enumerate(tasks):
        for l_idx, T in enumerate(lengths):
            for rep in (range(8) if is_instruct else (0, 1, 4, 6)):
                split_tag = "dev" if rep < 2 else "heldout_test"
                entity = f"SERVICE_{chr(65 + t_idx)}{l_idx}{rep}"
                v_gold = _make_identifier(inst_global, 0)
                v_stale = _make_identifier(inst_global, 1)
                assert v_gold != v_stale
                forbidden = [
                    v_gold,
                    v_stale,
                    v_gold.split("-")[0],
                    v_stale.split("-")[0],
                    v_gold.split("-")[1],
                    v_stale.split("-")[1],
                ]

                if is_instruct:
                    sys_prefix = (
                        "<|im_start|>system\n"
                        "You are a precise configuration state assistant. Reply ONLY with the exact active identifier "
                        "string (format word-number-word) and nothing else.<|im_end|>\n"
                        "<|im_start|>user\n"
                    )
                    query_str = (
                        f"Based solely on the surviving active records above, what is the active identifier for {entity}? "
                        "Output only the identifier.<|im_end|>\n"
                        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    )
                else:
                    sys_prefix = (
                        "Example 1:\n[ACTIVE] SERVICE_X00 identifier = pearl-9901-anchor\n"
                        "Question: Active identifier for SERVICE_X00?\nAnswer: pearl-9901-anchor\n\n"
                        "Example 2:\n[ACTIVE] SERVICE_Y00 identifier = topaz-9902-ledger\n"
                        "Question: Active identifier for SERVICE_Y00?\nAnswer: topaz-9902-ledger\n\n"
                        "Workspace Log:\n"
                    )
                    query_str = f"Question: Active identifier for {entity}?\nAnswer: "

                prefix_head_ids = tok.encode_plain(sys_prefix)
                gold_fact_ids = tok.encode_plain(
                    f"[AUTHORITATIVE RECORD] {entity} active identifier = {v_gold}\n"
                )
                stale_span1_ids = tok.encode_plain(
                    f"[TEMPORARY OVERRIDE] {entity} active identifier = {v_stale}\n"
                )
                stale_span2_ids = tok.encode_plain(
                    f"[DUPLICATE OVERRIDE] Confirming {entity} active identifier = {v_stale}\n"
                )
                stale_span3_ids = tok.encode_plain(
                    f"[CACHED OVERRIDE] Reminder that {entity} active identifier = {v_stale}\n"
                )
                query_ids = tok.encode_plain(query_str)

                spans: list[tuple[int, int]] = []
                sub_regime = ""

                if task_name == "TaskA_FactRetract":
                    is_multispan = rep >= 4
                    sub_regime = "multispan_retract" if is_multispan else "singlespan_retract"
                    p_pad = _build_filler_lines(tok, T // 5, inst_global * 11 + 1, forbidden)
                    cur = prefix_head_ids + gold_fact_ids + p_pad
                    s1 = len(cur)
                    cur = cur + stale_span1_ids
                    e1 = len(cur) - 1
                    spans.append((s1, e1))
                    if is_multispan:
                        mid_pad = _build_filler_lines(tok, T // 6, inst_global * 11 + 2, forbidden)
                        cur = cur + mid_pad
                        s2 = len(cur)
                        cur = cur + stale_span2_ids
                        e2 = len(cur) - 1
                        spans.append((s2, e2))
                        if rep >= 6:
                            mid_pad2 = _build_filler_lines(tok, T // 8, inst_global * 11 + 3, forbidden)
                            cur = cur + mid_pad2
                            s3 = len(cur)
                            cur = cur + stale_span3_ids
                            e3 = len(cur) - 1
                            spans.append((s3, e3))
                    rem = T - len(cur)
                    assert rem >= 64, f"Insufficient suffix room: rem={rem}"
                    cur = cur + _build_filler_lines(tok, rem, inst_global * 11 + 4, forbidden)
                    body_ids = cur

                elif task_name == "TaskB_RAGEvict":
                    rag_stale_a = tok.encode_plain(
                        f"[TOOL_OUTPUT Passage_Evict_A] Deprecated {entity} active identifier = {v_stale}.\n"
                    )
                    rag_stale_b = tok.encode_plain(
                        f"[TOOL_OUTPUT Passage_Evict_B] Superseded {entity} active identifier = {v_stale}.\n"
                    )
                    rag_gold = tok.encode_plain(
                        f"[TOOL_OUTPUT Passage_Keep_Gold] Verified current {entity} active identifier = {v_gold}.\n"
                    )
                    if rep in (0, 2, 4):
                        # Exact length-1 surviving island (\ell_r = 1) testing Theorem 7!
                        sub_regime = "short_island_len1"
                        lead = _build_filler_lines(tok, T // 5, inst_global * 13 + 1, forbidden)
                        cur = prefix_head_ids + rag_gold + lead
                        s1 = len(cur)
                        cur = cur + rag_stale_a
                        e1 = len(cur) - 1
                        cur = cur + island1_ids  # EXACT 1 surviving token!
                        s2 = len(cur)
                        cur = cur + rag_stale_b
                        e2 = len(cur) - 1
                        spans = [(s1, e1), (s2, e2)]
                        assert s2 - e1 - 1 == 1
                    elif rep in (1, 3, 5):
                        # Exact length-2 surviving island (\ell_r = 2) testing Theorem 7!
                        sub_regime = "short_island_len2"
                        lead = _build_filler_lines(tok, T // 5, inst_global * 13 + 2, forbidden)
                        cur = prefix_head_ids + rag_gold + lead
                        s1 = len(cur)
                        cur = cur + rag_stale_a
                        e1 = len(cur) - 1
                        cur = cur + island2_ids  # EXACT 2 surviving tokens!
                        s2 = len(cur)
                        cur = cur + rag_stale_b
                        e2 = len(cur) - 1
                        spans = [(s1, e1), (s2, e2)]
                        assert s2 - e1 - 1 == 2
                    else:
                        # Intentional C=256 chunk-boundary 3-token halo spill (e_1 = 511 -> spills into [512..514])
                        # plus gold passage surviving as a medium island between two evicted passages!
                        sub_regime = "chunk_boundary_halo_spill"
                        target_e1 = 511 if rep == 6 else 510  # inside [512-3, 511] for C=256!
                        base_len = len(prefix_head_ids) + len(rag_stale_a)
                        pad_before = (target_e1 + 1) - base_len
                        assert pad_before >= 16
                        lead = _build_filler_lines(tok, pad_before, inst_global * 13 + 3, forbidden)
                        cur = prefix_head_ids + lead
                        s1 = len(cur)
                        cur = cur + rag_stale_a
                        e1 = len(cur) - 1
                        assert e1 == target_e1
                        # Surviving gold passage right after boundary spill!
                        cur = cur + rag_gold
                        s2 = len(cur)
                        cur = cur + rag_stale_b
                        e2 = len(cur) - 1
                        spans = [(s1, e1), (s2, e2)]
                    rem = T - len(cur)
                    assert rem >= 64
                    body_ids = cur + _build_filler_lines(tok, rem, inst_global * 13 + 4, forbidden)

                else:
                    # Task C: Pinned-System-Prompt + Gold Config Prefix, Interior Sliding-Window Scrubbing
                    sub_regime = "pinned_sliding_window"
                    pin_pad = _build_filler_lines(tok, 128, inst_global * 17 + 1, forbidden)
                    cur = prefix_head_ids + gold_fact_ids + pin_pad
                    pin_end = len(cur)
                    scrub_pad_a = _build_filler_lines(tok, T // 8, inst_global * 17 + 2, forbidden)
                    scrub_pad_b = _build_filler_lines(tok, T // 8, inst_global * 17 + 3, forbidden)
                    s1 = len(cur)
                    cur = cur + scrub_pad_a + stale_span1_ids + scrub_pad_b + stale_span2_ids
                    e1 = len(cur) - 1
                    spans = [(s1, e1)]
                    assert s1 >= pin_end
                    rem = T - len(cur)
                    assert rem >= 128
                    body_ids = cur + _build_filler_lines(tok, rem, inst_global * 17 + 4, forbidden)

                if len(body_ids) != T:
                    raise RuntimeError(f"Prompt length mismatch: got {len(body_ids)}, expected {T}")

                geom = build_survivor_geometry(spans, T, W=CONV_W)
                clean_ids = [body_ids[i] for i in geom["pi"].tolist()]

                # Anti-leakage & strict BPE roundtrip verification on both original and compacted survivor sequences!
                clean_text = tok.decode_plain(clean_ids)
                orig_text = tok.decode_plain(body_ids)
                re_clean = tok.encode_plain(clean_text)
                re_orig = tok.encode_plain(orig_text)
                if re_orig != body_ids or re_clean != clean_ids:
                    raise RuntimeError(
                        f"BPE_BOUNDARY_MISMATCH at inst {inst_global}: "
                        f"orig={len(body_ids)}->{len(re_orig)}, clean={len(clean_ids)}->{len(re_clean)}"
                    )
                if sub_regime == "short_island_len1":
                    assert geom["island_lengths"] == [1] and len(geom["halo_indices"]) == 4
                if sub_regime == "short_island_len2":
                    assert geom["island_lengths"] == [2] and len(geom["halo_indices"]) == 5
                for C in (256, 1024):
                    plan_chymera_execution(T, C, spans)
                if v_stale.lower() in clean_text.lower():
                    raise RuntimeError(f"FILLER_LEAKAGE_ERROR: stale value {v_stale} survived deletion in inst {inst_global}!")
                if v_gold.lower() not in clean_text.lower():
                    raise RuntimeError(f"GOLD_MISSING_ERROR: gold value {v_gold} missing from clean_text in inst {inst_global}!")
                if v_stale.lower() not in orig_text.lower():
                    raise RuntimeError(f"STALE_MISSING_ERROR: stale value {v_stale} missing from orig_text in inst {inst_global}!")

                dataset.append(
                    {
                        "instance_id": inst_global,
                        "task": task_name,
                        "sub_regime": sub_regime,
                        "split": split_tag,
                        "T": T,
                        "entity": entity,
                        "v_gold": v_gold,
                        "v_stale": v_stale,
                        "P_pin": spans[0][0] if task_name == "TaskC_SlidingScrub" else None,
                        "W_keep": spans[0][1] + 1 if task_name == "TaskC_SlidingScrub" else None,
                        "spans": geom["canonical_spans"],
                        "island_lengths": geom["island_lengths"],
                        "halo_size": len(geom["halo_indices"]),
                        "L_prime": geom["L_prime"],
                        "body_ids": body_ids,
                        "query_ids": query_ids,
                        "body_sha256": hashlib.sha256(json.dumps(body_ids).encode()).hexdigest(),
                    }
                )
                inst_global += 1

    if is_instruct:
        assert len(dataset) == 72, f"Expected 72 instruct instances, got {len(dataset)}"
        assert sum(r["split"] == "dev" for r in dataset) == 18, "SPLIT_DEV_COUNT_MISMATCH"
        assert sum(r["split"] == "heldout_test" for r in dataset) == 54, "SPLIT_HELDOUT_COUNT_MISMATCH"
        assert sum(
            r["split"] == "heldout_test" and r["task"] == "TaskB_RAGEvict" and r["sub_regime"] == "short_island_len1"
            for r in dataset
        ) == 6, "HELDOUT_TASKB_LEN1_COUNT_MISMATCH"
        assert sum(
            r["split"] == "heldout_test" and r["task"] == "TaskB_RAGEvict" and r["sub_regime"] == "short_island_len2"
            for r in dataset
        ) == 6, "HELDOUT_TASKB_LEN2_COUNT_MISMATCH"
        assert sum(
            r["split"] == "heldout_test" and r["task"] == "TaskC_SlidingScrub" and r["T"] == 4096
            for r in dataset
        ) == 6, "HELDOUT_TASKC_T4096_COUNT_MISMATCH"
    else:
        assert len(dataset) == 24, f"Expected 24 base instances, got {len(dataset)}"

    if max_instances is not None:
        return dataset[:max_instances]
    return dataset


# ---------------------------------------------------------------------------
# Scoring & Statistical Evaluation Helpers
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Scoring only: trim outer whitespace/backticks, take first non-empty line."""
    txt = text.strip().strip("`").strip()
    for stop in ("<|im_end|>", "<|endoftext|>"):
        txt = txt.split(stop, 1)[0]
    return next((line.strip().strip("`").strip() for line in txt.splitlines() if line.strip()), "")


def token_f1(pred: str, gold: str) -> float:
    p_toks = [x for x in normalize_answer(pred).lower().replace("-", " - ").split() if x]
    g_toks = [x for x in normalize_answer(gold).lower().replace("-", " - ").split() if x]
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = 0
    g_counts: dict[str, int] = {}
    for t in g_toks:
        g_counts[t] = g_counts.get(t, 0) + 1
    for t in p_toks:
        if g_counts.get(t, 0) > 0:
            common += 1
            g_counts[t] -= 1
    if common == 0:
        return 0.0
    prec = common / len(p_toks)
    rec = common / len(g_toks)
    return 2.0 * prec * rec / (prec + rec)


def paired_bootstrap_ci(diffs: list[float], n_boot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def spearman_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(x) != len(y):
        return 0.0
    def ranks(values):
        a = np.asarray(values, dtype=np.float64)
        order = np.argsort(a, kind="stable")
        out = np.empty(len(a), dtype=np.float64)
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a[order[j]] == a[order[i]]:
                j += 1
            out[order[i:j]] = (i + j - 1) / 2
            i = j
        return out
    ax, ay = ranks(x), ranks(y)
    sx = ax - ax.mean()
    sy = ay - ay.mean()
    denom = float(np.sqrt(np.sum(sx * sx) * np.sum(sy * sy)))
    return float(np.sum(sx * sy) / denom) if denom > 0 else 0.0


def spearman_permutation_p_one_sided(x: list[float], y: list[float], n_perm: int = 2000, seed: int = SEED) -> float | None:
    """Monte Carlo randomization / permutation p-value (B=2000, plus-one correction) under H0: x independent of y given fixed S_+."""
    if len(x) < 3 or len(x) != len(y):
        return None
    rho_obs = spearman_corr(x, y)
    # Require non-constant ranks on both vectors.
    if np.all(np.asarray(x) == x[0]) or np.all(np.asarray(y) == y[0]):
        return None
    yv = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ge_obs = 0
    for _ in range(n_perm):
        y_perm = rng.permuted(yv)
        if spearman_corr(x, y_perm.tolist()) >= rho_obs:
            ge_obs += 1
    return float((1 + ge_obs) / (n_perm + 1))


def sign_flip_permutation_p_one_sided(diffs: list[float], n_perm: int = 2000, seed: int = SEED) -> float | None:
    """Monte Carlo randomization / permutation p-value (B=2000, plus-one correction) via Rademacher sign-flips under H0: symmetric around 0."""
    arr = np.asarray(diffs, dtype=np.float64)
    if not len(arr):
        return None
    obs = float(arr.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_perm, len(arr)))
    null_means = (signs * arr[None, :]).mean(axis=1)
    return float((1 + int(np.sum(null_means >= obs))) / (n_perm + 1))


# ---------------------------------------------------------------------------
# Exact Byte & FLOP Complexity Calculator (Theorem 6 Verification)
# ---------------------------------------------------------------------------


def compute_exact_memory_bytes(cfg_dict: dict, T: int, C: int, archive_dtype_bytes: int = 4) -> dict:
    """R2.13 payload, mathematical 3-slot ring AND DynamicCache's 4 slots.

    Delta16 quantizes matrices only; conv checkpoints always use BF16 here.
    Measured allocation (including any diagnostic tensors) is logged separately.
    """
    L_fa = cfg_dict["layers"] // 4
    L_gdn = cfg_dict["layers"] - L_fa
    hk, hv, hkv = (cfg_dict[k] for k in ("h_k", "h_v", "h_kv"))
    dk, dv, df = (cfg_dict.get(k, v) for k, v in (("d_k", 128), ("d_v", 128), ("d_fa", 256)))
    conv_dim = 2 * hk * dk + hv * dv
    m = len(partition_macro_chunks(T, C)) - 1
    kv = T * L_fa * 2 * hkv * df * 2
    J = L_gdn * hv
    ledgers = {}
    for slots, label in ((CONV_W - 1, "mathematical"), (CONV_W, "dynamic_cache")):
        ring = 2 * L_gdn * conv_dim * slots
        variants = {}
        for name, width, direct in (("Delta32", 4, False), ("Delta16", 2, False),
                                    ("Direct32", 4, True), ("Direct16", 2, True)):
            payload = m * (width * J * (dk * dk + (2 if direct else 1) * dk * dv) + ring)
            variants[name] = {"bytes": payload, "mib": payload / 2**20,
                              "extra_archive_over_kv_ratio": payload / kv if kv else 0.0}
        ledgers[label] = variants
    chosen = "32" if archive_dtype_bytes == 4 else "16"
    matrix = J * dk * dv * archive_dtype_bytes
    tree_ops = T * L_gdn * (2 * conv_dim + 8 * hv)
    tree = (2 * math.ceil(T / 16) - 1) * 4 * J * (dk * dk + dk * dv) + tree_ops
    return {"T": T, "C": C, "N_chunks": m + 1, "full_attn_kv_bytes": kv,
            "full_attn_kv_mib": kv / 2**20,
            "terminal_gdn_mib": L_gdn * (hv * dk * dv * 4 + conv_dim * CONV_W * 2) / 2**20,
            "mathematical": ledgers["mathematical"], "dynamic_cache": ledgers["dynamic_cache"],
            "chymera_delta_archive_mib": ledgers["mathematical"]["Delta" + chosen]["mib"],
            "chymera_direct_archive_mib": ledgers["mathematical"]["Direct" + chosen]["mib"],
            "extra_archive_over_kv_ratio": ledgers["mathematical"]["Delta" + chosen]["extra_archive_over_kv_ratio"],
            "clm_snapshot_archive_mib": m * (matrix + 2 * L_gdn * conv_dim * 3) / 2**20,
            "naive_tree_c16_mib": tree / 2**20, "per_token_archive_bytes": 0}


# ---------------------------------------------------------------------------
# Local CPU Dry-Run & Theorem Self-Test (`--cpu-self-test`)
# ---------------------------------------------------------------------------


def run_cpu_self_test() -> None:
    print("[self-test] Starting deterministic CPU dry-run & Theorem 6/7/8 verification...", flush=True)
    rng = np.random.default_rng(SEED)

    # 1) Load real Qwen3.5-0.8B tokenizer.json if available via `tokenizers` or pure JSON fallback check
    tok_path = Path(__file__).resolve().parent / "research_sources" / "qwen35_08b_inst" / "tokenizer.json"
    tok_obj = load_local_tokenizer(tok_path)
    print("[self-test] Loaded real tokenizer.json BPE", flush=True)

    # Verify all 72 Instruct prompts AND Base prompts
    ds_inst = build_benchmark_dataset(tok_obj, is_instruct=True)
    ds_base = build_benchmark_dataset(tok_obj, is_instruct=False, max_instances=24)
    assert len(ds_inst) == 72, f"Expected 72 instruct prompts, got {len(ds_inst)}"
    assert len(ds_base) == 24, f"Expected 24 base prompts, got {len(ds_base)}"

    # Verify CHymERA execution planner across all 72 prompts & strides C in {128, 256, 512}
    len1_count = 0
    len2_count = 0
    spill_count = 0
    for row in ds_inst:
        if row["sub_regime"] == "short_island_len1":
            assert row["island_lengths"] == [1]
            assert row["halo_size"] == 4  # {k1, k2, k2+1, k2+2} = 4 < 6 = 3*2!
            len1_count += 1
        elif row["sub_regime"] == "short_island_len2":
            assert row["island_lengths"] == [2]
            assert row["halo_size"] == 5  # {k1, k1+1, k2, k2+1, k2+2} = 5 < 6 = 3*2!
            len2_count += 1
        elif row["sub_regime"] == "chunk_boundary_halo_spill":
            spill_count += 1
        for C in (128, 256, 512, 1024):
            plan = plan_chymera_execution(row["T"], C, row["spans"], W=CONV_W)
            if row["sub_regime"] == "chunk_boundary_halo_spill" and C == 256:
                # Span 1 ends at 510 or 511 (inside chunk 1 = [256, 512)), so halo spills into chunk 2 = [512, 768)!
                assert plan["chunks"][2]["status"] == "dirty", "Chunk 2 must be marked dirty on boundary halo spill!"
            total_covered = plan["prefix_surv_len"] + sum(
                seg["surv_hi"] - seg["surv_lo"] for seg in plan["segments"]
            )
            assert total_covered == row["L_prime"], f"Survivor coverage mismatch: {total_covered} != {row['L_prime']}"
    assert len1_count == 9 and len2_count == 9 and spill_count == 6

    witness = plan_chymera_execution(48, 16, [(15, 29)])
    assert witness["replayed_tokens"] == 33
    for _ in range(500):
        T = int(rng.integers(4, 200))
        C = int(rng.integers(4, 64))
        points = sorted(rng.integers(0, T, size=2*int(rng.integers(1,5))).tolist())
        plan_chymera_execution(T, C, list(zip(points[::2],points[1::2])))

    # 2) Verify WY chunk folding & 2:1 GQA key-sharing shapes for BOTH 0.8B/2B (H_v=16) and 4B/9B (H_v=32, H_k=16)
    for model_tag in ("0.8B", "4B"):
        cfg_d = MODEL_REGISTRY[model_tag]["expected"]
        H_k, H_v = cfg_d["h_k"], cfg_d["h_v"]
        L_c = 140  # non-multiple of 64 to test zero-pad identity invariance!
        k_raw = rng.standard_normal((H_k, L_c, 128)).astype(np.float64)
        k_norm = k_raw / np.sqrt(np.sum(k_raw * k_raw, axis=-1, keepdims=True) + EPS)
        if H_v // H_k > 1:
            k_rep = np.repeat(k_norm, H_v // H_k, axis=0)
        else:
            k_rep = k_norm
        v = rng.standard_normal((H_v, L_c, 128)).astype(np.float64) * 0.3
        g = -np.abs(rng.standard_normal((H_v, L_c)).astype(np.float64)) * 0.05
        beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((H_v, L_c)).astype(np.float64)))
        S_init = rng.standard_normal((H_v, 128, 128)).astype(np.float64) * 0.2

        P_c, H_c = np_wy_chunk_summary(k_rep, v, g, beta, sub_chunk=64)
        S_wy = np.matmul(P_c, S_init) + H_c

        # Sequential ground truth
        S_seq = S_init.copy()
        for t in range(L_c):
            gam_t = np.exp(g[:, t])[:, None, None]
            S_dec = S_seq * gam_t
            mem = np.sum(S_dec * k_rep[:, t, :, None], axis=1)
            delta = (v[:, t] - mem) * beta[:, t, None]
            S_seq = S_dec + k_rep[:, t, :, None] * delta[:, None, :]

        rel_err = float(np.linalg.norm(S_wy - S_seq) / np.linalg.norm(S_seq))
        assert rel_err < 1e-13, f"WY folding mismatch on {model_tag}: rel_err={rel_err}"
        mem_info = compute_exact_memory_bytes(cfg_d, T=4096, C=256, archive_dtype_bytes=2)
        assert mem_info["per_token_archive_bytes"] == 0
        assert mem_info["chymera_delta_archive_mib"] < mem_info["naive_tree_c16_mib"] / 50.0

    # 3) Verify Theorem 8 Adjoint Readout Identity & Tightness on structured retrieval subspace (r_sub=8)
    m_steps = 96
    U_ret, _ = np.linalg.qr(rng.standard_normal((128, 8)))
    coeffs = rng.standard_normal((m_steps, 8))
    k_base = coeffs @ U_ret.T + 0.15 * rng.standard_normal((m_steps, 128))
    k_base /= np.linalg.norm(k_base, axis=-1, keepdims=True)
    q_T = U_ret @ rng.standard_normal(8)
    q_T /= np.linalg.norm(q_T) * np.sqrt(128.0)
    k_pert = k_base + 0.01 * rng.standard_normal((m_steps, 128))
    k_pert /= np.linalg.norm(k_pert, axis=-1, keepdims=True)
    v1 = rng.standard_normal((m_steps, 128)) * 0.2
    v2 = v1 + 0.01 * rng.standard_normal((m_steps, 128))
    g1 = -np.full(m_steps, 0.005)
    g2 = g1 + 0.0005 * rng.standard_normal(m_steps)
    b1 = np.full(m_steps, 0.75)
    b2 = np.clip(b1 + 0.01 * rng.standard_normal(m_steps), 0.05, 0.95)
    S0_1 = rng.standard_normal((128, 128)) * 0.1
    S0_2 = S0_1 + 0.02 * rng.standard_normal((128, 128))

    adj_res = np_adjoint_readout_check(k_base, v1, g1, b1, k_pert, v2, g2, b2, q_T, S0_1, S0_2)
    assert adj_res["identity_rel_err"] < 1e-13, f"Adjoint identity error {adj_res['identity_rel_err']}"
    assert adj_res["violation_dir"] <= 0.0 and adj_res["violation_cauchy"] <= 0.0
    assert adj_res["tightness_dir_vs_frob"] > 20.0, f"Got tightness={adj_res['tightness_dir_vs_frob']}"

    for tag in ("0.8B", "2B", "4B", "9B"):
        small = tag in ("0.8B", "2B")
        for C, div in ((256, 1), (1024, 5)):
            mem = compute_exact_memory_bytes(MODEL_REGISTRY[tag]["expected"], 4096, C)
            assert mem["full_attn_kv_mib"] == (48 if small else 128)
            for name, expected in (("Delta32", 549.4921875 if small else 1456.875),
                                   ("Delta16", 279.4921875 if small else 736.875),
                                   ("Direct32", 819.4921875 if small else 2176.875)):
                assert mem["mathematical"][name]["mib"] == expected / div
            assert mem["dynamic_cache"]["Delta32"]["mib"] > mem["mathematical"]["Delta32"]["mib"]
    assert normalize_answer("\n  `amber-1234-ledger`\nextra") == "amber-1234-ledger"
    run_tiny_arch_self_test()
    print("[self-test] PASS: 72/72 Instruct + 24/24 Base BPE, geometry, WY, adjoint, memory; 0 failures", flush=True)


# ---------------------------------------------------------------------------
# GPU Execution Engine (`Qwen3.5` Real-Weight 7-Method Evaluation)
# ---------------------------------------------------------------------------


def ensure_transformers_installed(allow_install: bool) -> None:
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qm  # noqa: F401

        sha = hashlib.sha256(Path(qm.__file__).read_bytes()).hexdigest()
        if sha == LOCKED_MODELING_SHA256:
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


class JsonBPETokenizer:
    """Offline execution of the real byte-level BPE for this ASCII prompt suite.

    Uses the actual vocabulary, ranked merges and added tokens, never mock IDs.
    Non-ASCII text requires the Rust tokenizers implementation (fail closed).
    """
    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        self.vocab = data["model"]["vocab"]
        self.vocab_size = len(self.vocab)
        self.reverse = {v: k for k, v in self.vocab.items()}
        self.ranks = {tuple(x.split(" ") if isinstance(x, str) else x): i
                      for i, x in enumerate(data["model"]["merges"])}
        self.special = {x["content"]: x["id"] for x in data["added_tokens"]}
        self.reverse_special = {v: k for k, v in self.special.items()}
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.byte_chars = dict(zip(bs, map(chr, cs)))
        self.char_bytes = {c: b for b, c in self.byte_chars.items()}
        self.pattern = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
        self.special_pattern = re.compile("(" + "|".join(re.escape(x) for x in sorted(self.special, key=len, reverse=True)) + ")")
        self.eos_token_id = 248044

    @lru_cache(maxsize=32768)
    def _bpe(self, text):
        parts = [self.byte_chars[b] for b in text.encode("utf-8")]
        while len(parts) > 1:
            candidates = [(self.ranks.get((parts[i], parts[i + 1]), math.inf), i)
                          for i in range(len(parts) - 1)]
            rank, i = min(candidates)
            if math.isinf(rank):
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return tuple(self.vocab[p] for p in parts)

    def encode(self, text, add_special_tokens=False):
        text = unicodedata.normalize("NFC", text)
        ids = []
        for segment in self.special_pattern.split(text):
            if segment in self.special:
                ids.append(self.special[segment])
            elif segment:
                if not segment.isascii():
                    raise ValueError("Install tokenizers for non-ASCII BPE")
                for word in self.pattern.findall(segment):
                    ids.extend(self._bpe(word))
        return ids

    def decode(self, ids, skip_special_tokens=False):
        out, pending = [], bytearray()
        for i in ids:
            if i in self.reverse_special:
                out.append(pending.decode("utf-8")); pending.clear()
                if not skip_special_tokens:
                    out.append(self.reverse_special[i])
            else:
                pending.extend(self.char_bytes[c] for c in self.reverse[i])
        out.append(pending.decode("utf-8"))
        return "".join(out)


def load_local_tokenizer(path):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return JsonBPETokenizer(path)
    return Tokenizer.from_file(str(path))


def write_json(path, value):
    """Atomic checkpoint: readers never see a half-written summary."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(sanitize_json(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(sanitize_json(row), allow_nan=False) + "\n")
    tmp.replace(path)


def tau_name(tau):
    return f"CHymERA-Certified({tau})"


def closure_for_tau(scores, tau):
    if tau == 0:
        return list(range(len(scores)))
    hits = [b for b, score in enumerate(scores) if score > tau]
    return list(range(max(hits) + 1)) if hits else []


def torch_wy_chunk_summary(key, value, g, beta, use_qk_l2norm=True, sub_chunk=64):
    """FP32 UT-WY, same normalized operands and 64-token folding as Qwen."""
    import torch
    import torch.nn.functional as F
    if not bool(key.isfinite().all() and value.isfinite().all() and g.isfinite().all() and beta.isfinite().all()):
        raise AssertionError("NONFINITE_WY_OPERANDS")
    if key.shape[2] < value.shape[2]:
        key = key.repeat_interleave(value.shape[2] // key.shape[2], dim=2)
    k, v = key[0].transpose(0, 1).float(), value[0].transpose(0, 1).float()
    b, dec = beta[0].T.float(), g[0].T.float()
    if use_qk_l2norm:
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + EPS)
    hv, n, dk = k.shape
    dv = v.shape[-1]
    pad = (-n) % sub_chunk
    k, v = F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
    b, dec = F.pad(b, (0, pad)), F.pad(dec, (0, pad))
    ns = (n + pad) // sub_chunk
    k, v = k.reshape(hv, ns, sub_chunk, dk), v.reshape(hv, ns, sub_chunk, dv)
    b, dec = b.reshape(hv, ns, sub_chunk), dec.reshape(hv, ns, sub_chunk)
    kb, vb = k * b[..., None], v * b[..., None]
    cg = dec.cumsum(-1)
    upper = torch.ones(sub_chunk, sub_chunk, dtype=torch.bool, device=k.device).triu(1)
    decay = (cg[..., None] - cg[..., None, :]).masked_fill(upper, -torch.inf).exp()
    system = (kb @ k.transpose(-1, -2)) * decay
    u = torch.linalg.solve_triangular(system, vb, upper=False, unitriangular=True)
    y = torch.linalg.solve_triangular(system, kb * cg.exp()[..., None], upper=False, unitriangular=True)
    kend = k * (cg[..., -1:] - cg).exp()[..., None]
    P = torch.eye(dk, device=k.device).expand(hv, dk, dk).clone()
    H = torch.zeros(hv, dk, dv, device=k.device)
    for i in range(ns):
        kt, gm = kend[:, i].transpose(-1, -2), cg[:, i, -1].exp()[:, None, None]
        P = gm * P - torch.bmm(kt, torch.bmm(y[:, i], P))
        H = gm * H + torch.bmm(kt, u[:, i] - torch.bmm(y[:, i], H))
    return P, H


class HybridBenchEngine:
    """Shared real Qwen execution helpers, also exercised by CPU tiny-arch runs."""
    def __init__(self, full, tokenizer, device):
        import torch
        from transformers.cache_utils import DynamicCache
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
        class BenchCache(DynamicCache):
            # Pre-lock transformers releases expose only the layer containers.
            # Keep the CPU architecture test usable there; GPU SHA gate is strict.
            def is_recording_past(self, layer_idx):
                return self.layers[layer_idx].record_past
            def get_recurrent_state(self, layer_idx, state_idx=0):
                layer = self.layers[layer_idx]
                return layer.recurrent_states[state_idx] if layer.has_previous_state[state_idx] else None
            def get_conv_state(self, layer_idx, state_idx=0):
                layer = self.layers[layer_idx]
                return layer.conv_states[state_idx] if layer.has_previous_state[state_idx] else None
        self.torch, self.qm, self.DynamicCache = torch, qm, BenchCache
        self.model = full.model.language_model
        self.lm_head, self.cfg = full.lm_head, full.config.text_config
        self.tokenizer, self.device = tokenizer, device
        self.gdn = [i for i, t in enumerate(self.cfg.layer_types) if t == "linear_attention"]
        self.fa = [i for i, t in enumerate(self.cfg.layer_types) if t == "full_attention"]
        self.n_blocks = self.cfg.num_hidden_layers // 4
        self.orig_chunk_fn, self.orig_recur_fn = qm.torch_chunk_gated_delta_rule, qm.torch_recurrent_gated_delta_rule
        self.mode, self.layer, self.phase = "off", None, None
        self._capture_offload_cpu = False
        self.h, self.ops, self.ph, self.q, self.gamma = {}, {}, {}, {}, {}
        self.f_oracle_buffer = None
        self.clean_suffix_fa_mlp_calls = 0
        self.unrefreshed_block_fa_mlp_calls = 0
        self.closure_cutoff_layer = 0
        self.startup_closure_checks = {}
        self.jump_max_abs_err = 0.0
        self.jump_max_rel_err = 0.0
        self.handles = []
        for l in self.gdn:
            def pre(mod, args, kwargs, l=l):
                self.layer = l
                if self.mode in ("capture", "capture_h"):
                    inp = (args[0] if args else kwargs["hidden_states"]).detach()
                    self.h[l] = inp.cpu() if self._capture_offload_cpu else inp.clone()
            def post(mod, args, out):
                self.layer = None
            self.handles.append(self.model.layers[l].linear_attn.register_forward_pre_hook(pre, with_kwargs=True))
            self.handles.append(self.model.layers[l].linear_attn.register_forward_hook(post))
        for l, layer in enumerate(self.model.layers):
            def count(mod, args, l=l):
                if self.phase in ("clean_bmm", "replay_clean_suffix"):
                    self.clean_suffix_fa_mlp_calls += 1
                elif self.phase == "replay_upstream_closure":
                    self.clean_suffix_fa_mlp_calls += 1
                    if l >= self.closure_cutoff_layer:
                        self.unrefreshed_block_fa_mlp_calls += 1
            self.handles.append(layer.mlp.register_forward_pre_hook(count))
            if l in self.fa:
                self.handles.append(layer.self_attn.register_forward_pre_hook(count))
        qm.torch_chunk_gated_delta_rule = self.wrap(self.orig_chunk_fn)
        qm.torch_recurrent_gated_delta_rule = self.wrap(self.orig_recur_fn)

    def close(self):
        self.qm.torch_chunk_gated_delta_rule = self.orig_chunk_fn
        self.qm.torch_recurrent_gated_delta_rule = self.orig_recur_fn
        for h in self.handles:
            h.remove()

    def wrap(self, original):
        def fn(query, key, value, g, beta, initial_state=None, output_final_state=False,
               use_qk_l2norm_in_kernel=False, **kw):
            l, torch = self.layer, self.torch
            if l is not None:
                if self.mode == "wy":
                    self.ph[l] = torch_wy_chunk_summary(key, value, g, beta, use_qk_l2norm_in_kernel)
                    self.gamma[l] = g[0].double().sum(0).exp()
                if self.mode in ("capture", "capture_ops", "probe"):
                    k, q = key.float(), query.float()
                    if use_qk_l2norm_in_kernel:
                        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + EPS)
                        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + EPS)
                    self.q[l] = q[0, -1].detach() * q.shape[-1] ** -0.5
                    if self.mode in ("capture", "capture_ops"):
                        if self._capture_offload_cpu:
                            self.ops[l] = {"k": k[0].detach().cpu(), "v": value[0].float().detach().cpu(),
                                           "g": g[0].float().detach().cpu(), "beta": beta[0].float().detach().cpu()}
                        else:
                            self.ops[l] = {"k": k[0].detach(), "v": value[0].float().detach(),
                                           "g": g[0].float().detach(), "beta": beta[0].float().detach()}
            return original(query, key, value, g=g, beta=beta, initial_state=initial_state,
                            output_final_state=output_final_state,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)
        return fn

    def cache(self):
        return self.DynamicCache(config=self.cfg)

    def tensor(self, ids):
        return self.torch.tensor([list(ids)], dtype=self.torch.long, device=self.device)

    def positions(self, n, start=0):
        return self.torch.arange(start, start + n, device=self.device).unsqueeze(0)

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def timed(self, fn):
        self.sync(); t = time.perf_counter()
        value = fn()
        self.sync()
        return value, (time.perf_counter() - t) * 1000

    def forward(self, ids, cache=None, positions=None):
        return self.model(input_ids=ids, past_key_values=self.cache() if cache is None else cache, position_ids=positions, use_cache=True)

    def capture(self, ids, positions, cache=None, operands=True, offload_cpu=False, include_h=True, chunk_stride=None):
        self.mode = ("capture_ops" if (operands and not include_h) else "capture" if operands else "capture_h")
        self._capture_offload_cpu = bool(offload_cpu)
        self.h, self.ops = {}, {}
        try:
            if chunk_stride is None:
                out = self.forward(ids, cache, positions)
                return out.past_key_values, dict(self.h), dict(self.ops)
            work_cache = self.cache() if cache is None else cache
            chunks = partition_macro_chunks(ids.shape[1], int(chunk_stride))
            h_parts = {l: [] for l in self.gdn}
            ops_parts = {l: {"k": [], "v": [], "g": [], "beta": []} for l in self.gdn}
            for lo, hi in chunks:
                self.h, self.ops = {}, {}
                out = self.forward(ids[:, lo:hi], work_cache, positions[:, lo:hi])
                work_cache = out.past_key_values
                if include_h:
                    for l in self.gdn:
                        h_parts[l].append(self.h[l])
                if operands:
                    for l in self.gdn:
                        for k_name in ("k", "v", "g", "beta"):
                            ops_parts[l][k_name].append(self.ops[l][k_name])
                del out
            h_out = {l: self.torch.cat(h_parts[l], dim=1) for l in self.gdn} if include_h else {}
            ops_out = (
                {
                    l: {k_name: self.torch.cat(ops_parts[l][k_name], dim=0) for k_name in ("k", "v", "g", "beta")}
                    for l in self.gdn
                }
                if operands else {}
            )
            return work_cache, h_out, ops_out
        finally:
            self.mode = "off"
            self._capture_offload_cpu = False
            self.h, self.ops = {}, {}

    def build_archive(self, body, positions, C):
        cache, anchors, rings, ends, PH, Gamma = self.cache(), {}, {}, {}, {}, {}
        chunks = partition_macro_chunks(body.shape[1], C)
        for c, (lo, hi) in enumerate(chunks):
            if c:
                anchors[c] = {l: cache.get_recurrent_state(l, 0).clone() for l in self.gdn}
                rings[c] = {l: cache.get_conv_state(l, 0).clone() for l in self.gdn}
                ends[c - 1] = rings[c]  # shared storage, no duplicate ring archive
            self.mode, self.ph, self.gamma = ("wy" if c else "off"), {}, {}
            try:
                self.forward(body[:, lo:hi], cache, positions[:, lo:hi])
                if c:
                    PH[c] = dict(self.ph)
                    Gamma[c] = dict(self.gamma)
            finally:
                self.mode = "off"
        ends[len(chunks) - 1] = {l: cache.get_conv_state(l, 0) for l in self.gdn}
        return {"base_cache": cache, "anchor_S": anchors, "anchor_conv": rings,
                "end_conv": ends, "PH": PH, "Gamma": Gamma, "chunks": chunks, "C": C}

    def build_anchor_only_archive(self, body, positions, C):
        """Chunked prefill saving only (anchor_S, anchor_conv) without WY (P_c, H_c) folding."""
        cache, anchors, rings = self.cache(), {}, {}
        chunks = partition_macro_chunks(body.shape[1], C)
        self.mode = "off"
        for c, (lo, hi) in enumerate(chunks):
            if c:
                anchors[c] = {l: cache.get_recurrent_state(l, 0).clone() for l in self.gdn}
                rings[c] = {l: cache.get_conv_state(l, 0).clone() for l in self.gdn}
            self.forward(body[:, lo:hi], cache, positions[:, lo:hi])
        return {"base_cache": cache, "anchor_S": anchors, "anchor_conv": rings, "chunks": chunks, "C": C}

    def boundary_cache(self, archive, c, prefix_len=None):
        if c == 0:
            return self.cache()
        cache = copy.deepcopy(archive["base_cache"])
        n = archive["chunks"][c][0] if prefix_len is None else prefix_len
        for l in self.fa:
            cache.layers[l].keys = cache.layers[l].keys[:, :, :n].clone()
            cache.layers[l].values = cache.layers[l].values[:, :, :n].clone()
        for l in self.gdn:
            self.set_state(cache, l, archive["anchor_S"][c][l], archive["anchor_conv"][c][l])
        return cache

    def set_state(self, cache, l, S, conv):
        cache.update_recurrent_state(S.clone(), l)
        cache.update_conv_state(conv.clone(), l, conv_kernel_size=CONV_W)
        # update_conv_state concatenates when already initialized; overwrite the ring.
        cache.layers[l].conv_states[0].copy_(conv)
        cache.layers[l].has_previous_state[0] = True

    def evict(self, base, pi):
        cache = copy.deepcopy(base)
        for l in self.fa:
            cache.layers[l].keys = base.layers[l].keys.index_select(-2, pi).contiguous()
            cache.layers[l].values = base.layers[l].values.index_select(-2, pi).contiguous()
        return cache

    def counted_replay(self, fn, has_clean=True):
        self.clean_suffix_fa_mlp_calls = 0
        self.phase = "replay_clean_suffix" if has_clean else None
        try:
            return fn()
        finally:
            self.phase = None

    def sparse_replay(self, archive, plan, clean, pos):
        c, u = plan["first_non_clean_chunk"], plan["prefix_surv_len"]
        work = self.boundary_cache(archive, c, u)
        self.clean_suffix_fa_mlp_calls = 0
        if u < clean.shape[1]:
            self.counted_replay(lambda: self.forward(clean[:, u:], work, pos[:, u:]),
                                any(seg["kind"] == "clean_bmm" for seg in plan["segments"]))
        return work

    def advance_f_input(self, l, h, work):
        """Execute exact Qwen projection/conv/kernel without FA or MLP calls.

        Calling linear_attn itself uses the locked update_conv_state, fused single
        token branch, and chunk/recurrent kernel verbatim; no token Python loop.
        """
        if h.shape[1]:
            self.model.layers[l].linear_attn(h, cache_params=work)

    def jump(self, archive, work, seg, seam):
        torch = self.torch
        self.phase = "clean_bmm"
        try:
            first = seg["chunks"][0]
            for l in self.gdn:
                seam[l] = work.get_recurrent_state(l, 0)[0].clone() - archive["anchor_S"][first][l][0]
                S = work.get_recurrent_state(l, 0)[0]
                for c in seg["chunks"]:
                    P, H = archive["PH"][c][l]
                    direct = torch.bmm(P, S) + H
                    endS = (archive["anchor_S"][c + 1][l][0]
                            if c + 1 in archive["anchor_S"] else archive["base_cache"].get_recurrent_state(l, 0)[0])
                    delta = endS + torch.bmm(P, S - archive["anchor_S"][c][l][0])
                    err = float((direct - delta).abs().max())
                    s_scale = max(
                        1.0,
                        float(direct.abs().max()),
                        float(endS.abs().max()),
                        float(S.abs().max()),
                        float(archive["anchor_S"][c][l][0].abs().max()),
                    )
                    rel_err = err / s_scale
                    self.jump_max_abs_err = max(self.jump_max_abs_err, err)
                    self.jump_max_rel_err = max(self.jump_max_rel_err, rel_err)
                    if not math.isfinite(err) or err > 1e-4 * s_scale:
                        raise AssertionError(
                            f"DIRECT_DELTA_MISMATCH layer={l}, chunk={c}: abs={err}, rel={rel_err}, scale={s_scale}"
                        )
                    S = direct
                self.set_state(work, l, S.unsqueeze(0), archive["end_conv"][seg["chunks"][-1]][l])
        finally:
            self.phase = None

    def repair_F(self, archive, plan, body, orig_pos, pi):
        """Pass 1 original dirty-chunk h; pass 2 independent per-layer F stream."""
        assert self.f_oracle_buffer is None and not self.h and not self.ops, "F_ORACLE_ISOLATION_VIOLATION"
        torch = self.torch
        base_kv_ptrs = [archive["base_cache"].layers[l].keys.data_ptr() for l in self.fa]
        base_kv_lens = [archive["base_cache"].layers[l].keys.shape[-2] for l in self.fa]
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            mem_before = torch.cuda.memory_allocated(self.device)
        else:
            mem_before = 0
        self.sync(); t_p1 = time.perf_counter()
        dirty_h = {}
        replayed_orig = 0
        for rec in plan["chunks"]:
            if rec["status"] == "dirty":
                c, lo, hi = rec["c"], rec["orig_lo"], rec["orig_hi"]
                tmp = self.boundary_cache(archive, c)
                if c > 0:
                    l0 = self.gdn[0]
                    assert tmp.get_recurrent_state(l0, 0).data_ptr() != archive["base_cache"].get_recurrent_state(l0, 0).data_ptr()
                    assert float((tmp.get_recurrent_state(l0, 0) - archive["anchor_S"][c][l0]).abs().max()) == 0.0
                _, dirty_h[c], _ = self.capture(body[:, lo:hi], orig_pos[:, lo:hi], tmp, operands=False)
                replayed_orig += hi - lo
                del tmp
        self.sync(); pass1_ms = (time.perf_counter() - t_p1) * 1000.0
        assert [archive["base_cache"].layers[l].keys.data_ptr() for l in self.fa] == base_kv_ptrs, "BASE_CACHE_MUTATED_IN_PASS1"
        assert [archive["base_cache"].layers[l].keys.shape[-2] for l in self.fa] == base_kv_lens, "BASE_CACHE_LEN_MUTATED_IN_PASS1"
        transient_bytes = allocated_tensor_bytes(dirty_h)
        if self.device.type == "cuda":
            peak_transient_mib = max(transient_bytes, torch.cuda.max_memory_allocated(self.device) - mem_before) / (2**20)
        else:
            peak_transient_mib = transient_bytes / (2**20)
        self.sync(); t_p2 = time.perf_counter()
        c0, u = plan["first_non_clean_chunk"], plan["prefix_surv_len"]
        work = self.boundary_cache(archive, c0, u)
        seam = {}
        self.clean_suffix_fa_mlp_calls = 0
        for seg in plan["segments"]:
            if seg["kind"] == "dirty_replay":
                for l in self.gdn:
                    hs = []
                    for c in seg["chunks"]:
                        rec = plan["chunks"][c]
                        ix = pi[rec["surv_lo"]:rec["surv_hi"]] - rec["orig_lo"]
                        hs.append(dirty_h[c][l].index_select(1, ix))
                    self.advance_f_input(l, self.torch.cat(hs, dim=1), work)
            else:
                self.jump(archive, work, seg, seam)
        # Only terminal clean seam enters R2.35.
        if not plan["segments"] or plan["segments"][-1]["kind"] != "clean_bmm":
            seam = {}
        assert self.clean_suffix_fa_mlp_calls == 0
        self.sync(); pass2_ms = (time.perf_counter() - t_p2) * 1000.0
        del dirty_h
        # Pass 1 finished before surviving FA KV eviction.
        self.sync(); t_ev = time.perf_counter()
        for l in self.fa:
            work.layers[l] = copy.deepcopy(archive["base_cache"].layers[l])
            work.layers[l].keys = work.layers[l].keys.index_select(-2, pi).contiguous()
            work.layers[l].values = work.layers[l].values.index_select(-2, pi).contiguous()
        self.sync(); kv_evict_ms_F = (time.perf_counter() - t_ev) * 1000.0
        audit_meta = {
            "f_oracle_buffer_was_none": True,
            "base_cache_unmutated": True,
            "pass1_ms": pass1_ms,
            "pass2_ms": pass2_ms,
            "kv_evict_ms": kv_evict_ms_F,
            "pass1_replayed_orig_tokens": replayed_orig,
            "peak_transient_mib": peak_transient_mib,
        }
        return work, seam, audit_meta

    def repair_fast(self, archive, plan, clean, pos, pi):
        c, u = plan["first_non_clean_chunk"], plan["prefix_surv_len"]
        work = self.boundary_cache(archive, c, u)
        seam = {}
        self.clean_suffix_fa_mlp_calls = 0
        for seg in plan["segments"]:
            lo, hi = seg["surv_lo"], seg["surv_hi"]
            if seg["kind"] == "dirty_replay":
                self.forward(clean[:, lo:hi], work, pos[:, lo:hi])
            else:
                self.jump(archive, work, seg, seam)
                self.phase = "clean_bmm"
                try:
                    for l in self.fa:
                        src = archive["base_cache"].layers[l]
                        ix = pi[lo:hi]
                        # A clean first segment can follow fully dropped prefix chunks.
                        k, v = src.keys.index_select(-2, ix), src.values.index_select(-2, ix)
                        work.update(k, v, l)
                finally:
                    self.phase = None
        if not plan["segments"] or plan["segments"][-1]["kind"] != "clean_bmm":
            seam = {}
        assert self.clean_suffix_fa_mlp_calls == 0
        return work, seam

    def replay_upstream_blocks(self, b_star, fast, archive, plan, clean, positions):
        """Actually run the upstream closure; layers above it retain Fast cache."""
        self.clean_suffix_fa_mlp_calls = 0
        self.unrefreshed_block_fa_mlp_calls = 0
        if b_star == -1:
            return fast
        work = copy.deepcopy(fast)
        prefix = self.boundary_cache(archive, plan["first_non_clean_chunk"], plan["prefix_surv_len"])
        n_layers = 4 * (b_star + 1)
        for l in range(n_layers):
            work.layers[l] = copy.deepcopy(prefix.layers[l])
        u = plan["prefix_surv_len"]
        if u < clean.shape[1]:
            h = self.model.embed_tokens(clean[:, u:])
            text_pos = positions[:, u:]
            rope_pos = text_pos[None].expand(3, -1, -1)
            mask_kw = dict(config=self.cfg, inputs_embeds=h, attention_mask=None,
                           past_key_values=work, position_ids=text_pos)
            masks = {"full_attention": self.qm.create_causal_mask(**mask_kw),
                     "linear_attention": self.qm.create_recurrent_attention_mask(**mask_kw)}
            pe = self.model.rotary_emb(h, rope_pos)
            self.closure_cutoff_layer = n_layers
            self.phase = "replay_upstream_closure" if any(seg["kind"] == "clean_bmm" for seg in plan["segments"]) else None
            try:
                for l, layer in enumerate(self.model.layers[:n_layers]):
                    h = layer(h, position_embeddings=pe, attention_mask=masks[self.cfg.layer_types[l]],
                              position_ids=text_pos, past_key_values=work, use_cache=True)
            finally:
                self.phase = None
            assert self.unrefreshed_block_fa_mlp_calls == 0, "UNREFRESHED_BLOCK_CALL_VIOLATION"
        return work

    def cache_difference(self, a, b, return_rel=False):
        diffs, rel_diffs = [], []
        for l in self.gdn:
            for getter in ("get_recurrent_state", "get_conv_state"):
                av, bv = getattr(a, getter)(l, 0), getattr(b, getter)(l, 0)
                d = float((av - bv).abs().max())
                sc = max(1.0, float(av.abs().max()), float(bv.abs().max()))
                diffs.append(d)
                rel_diffs.append(d / sc)
        for l in self.fa:
            for attr in ("keys", "values"):
                av, bv = getattr(a.layers[l], attr), getattr(b.layers[l], attr)
                assert av.shape == bv.shape
                if av.numel():
                    d = float((av - bv).abs().max())
                    sc = max(1.0, float(av.abs().max()), float(bv.abs().max()))
                else:
                    d, sc = 0.0, 1.0
                diffs.append(d)
                rel_diffs.append(d / sc)
        max_abs = max(diffs, default=0.0)
        max_rel = max(rel_diffs, default=0.0)
        return (max_abs, max_rel) if return_rel else max_abs

    def greedy_decode(self, cache, query, start_pos, max_new_tokens=24):
        """Only im_end/eos stop generation; first logits use identical query IDs."""
        eos_set = {248046}
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            eos_set.add(eos)
        assert 198 not in eos_set
        work = copy.deepcopy(cache)
        out = self.forward(query, work, self.positions(query.shape[1], start_pos))
        logits = self.lm_head(out.last_hidden_state[:, -1]).float()[0]
        first, ids, stopped = logits.detach().clone(), [], False
        for step in range(max_new_tokens):
            token = int(logits.argmax())
            if token in eos_set:
                stopped = True
                break
            ids.append(token)
            if step + 1 < max_new_tokens:
                out = self.forward(self.tensor([token]), work, self.positions(1, start_pos + query.shape[1] + step))
                logits = self.lm_head(out.last_hidden_state[:, -1]).float()[0]
        text = FastTokenizerAdapter(self.tokenizer).decode_plain(ids)
        return {"pred": normalize_answer(text), "first_logits": first, "generated_ids": ids,
                "truncated": not stopped and len(ids) == max_new_tokens}

    def probe(self, cache, query, T):
        self.mode, self.q = "probe", {}
        try:
            self.forward(query, copy.deepcopy(cache), self.positions(query.shape[1], T))
            return dict(self.q)
        finally:
            self.mode = "off"

    def frozen_scores(self, archive, plan, seam, seam_F, q):
        torch = self.torch
        names = ("eta_b_max", "eta_b_mean", "eta_b_frob", "eta_b_max_F")
        result = {n: [0.0] * self.n_blocks for n in names}
        if not seam:
            return result
        suffix = plan["segments"][-1]["chunks"]
        first = suffix[0]
        for b in range(self.n_blocks):
            scores, fscores, frobs = [], [], []
            for l in (4*b, 4*b+1, 4*b+2):
                p = q[l].unsqueeze(-1)
                gain = torch.ones_like(q[l].norm(dim=-1))
                for c in reversed(suffix):
                    P = archive["PH"][c][l][0]
                    p = torch.bmm(P.transpose(-1, -2), p)
                    gain *= archive["Gamma"][c][l].to(gain)
                # Reference is fixed PRE-refresh seam anchor, never terminal state.
                scale = archive["anchor_S"][first][l][0].norm(dim=(-2, -1)) + 1
                denom = q[l].norm(dim=-1).clamp_min(1e-12)
                a = p.squeeze(-1).norm(dim=-1) / denom
                scores.extend((a * seam[l].norm(dim=(-2,-1)) / scale).tolist())
                fscores.extend((a * seam_F[l].norm(dim=(-2,-1)) / scale).tolist())
                frobs.extend((gain * seam[l].norm(dim=(-2,-1)) / scale).tolist())
            result["eta_b_max"][b] = max(scores)
            result["eta_b_mean"][b] = float(np.mean(scores))
            result["eta_b_frob"][b] = max(frobs)
            result["eta_b_max_F"][b] = max(fscores)
        return result

    def operand_replay(self, ops, indices, initial=None):
        """Vectorized native chunk kernel on captured post-conv operands."""
        torch = self.torch
        k = ops["k"].index_select(0, indices).unsqueeze(0)
        v = ops["v"].index_select(0, indices).unsqueeze(0)
        g = ops["g"].index_select(0, indices).unsqueeze(0)
        beta = ops["beta"].index_select(0, indices).unsqueeze(0)
        if not indices.numel():
            return initial
        _, S = self.orig_chunk_fn(torch.zeros_like(k), k, v, g=g, beta=beta,
                                 initial_state=initial, output_final_state=True,
                                 use_qk_l2norm_in_kernel=False)
        return S

    def nohalo(self, archive, plan, pi, ops):
        work = self.evict(archive["base_cache"], pi)
        c0 = plan["first_non_clean_chunk"]
        for l in self.gdn:
            ops_l = {name: t.to(device=self.device) for name, t in ops[l].items()}
            S = (self.torch.zeros_like(work.get_recurrent_state(l, 0)) if c0 == 0
                 else archive["anchor_S"][c0][l].clone())
            for seg in plan["segments"]:
                if seg["kind"] == "dirty_replay":
                    S = self.operand_replay(ops_l, pi[seg["surv_lo"]:seg["surv_hi"]], S)
                else:
                    for c in seg["chunks"]:
                        P, H = archive["PH"][c][l]
                        S = (self.torch.bmm(P, S[0]) + H).unsqueeze(0)
            work.update_recurrent_state(S, l)
            del ops_l
        return work

    def f_oracle(self, archive, pi, full_h):
        self.f_oracle_buffer = full_h
        try:
            work = self.cache()
            pi_cpu = pi.detach().cpu()
            for l in self.gdn:
                h_src = full_h[l]
                h_l = (h_src.index_select(1, pi_cpu).to(device=self.device)
                       if h_src.device.type == "cpu" else h_src.index_select(1, pi))
                self.advance_f_input(l, h_l, work)
                del h_l
            for l in self.fa:
                work.layers[l] = copy.deepcopy(archive["base_cache"].layers[l])
                work.layers[l].keys = work.layers[l].keys.index_select(-2, pi)
                work.layers[l].values = work.layers[l].values.index_select(-2, pi)
            return work
        finally:
            self.f_oracle_buffer = None

    def naive_indices(self, spans, T):
        indices, end = [], 0
        for r, (s, e) in enumerate(spans):
            indices.extend(range(end, s))
            if r:
                # Independent original pre-span halo can resurrect deleted context.
                for t in range(max(0, s - 3), s):
                    if t not in indices[-3:]:
                        indices.append(t)
            end = e + 1
        indices.extend(range(end, T))
        return self.torch.tensor(indices, device=self.device, dtype=self.torch.long)

    def terminal_inverse(self, archive, pi, spans, ops):
        """Batched 3-layer-at-a-time reverse update across all GDN layers/value heads (VRAM-bounded for 9B)."""
        torch = self.torch
        work = self.evict(archive["base_cache"], pi)
        hv = self.cfg.linear_num_value_heads
        ix = pi[pi >= spans[0][0]]
        clamps = 0
        for g_start in range(0, len(self.gdn), 3):
            group_layers = self.gdn[g_start:g_start + 3]
            S = torch.cat([work.get_recurrent_state(l, 0)[0] for l in group_layers], dim=0).float()
            ref = S.norm(dim=(-2,-1), keepdim=True) + 1
            stacked = {name: torch.cat([ops[l][name].to(device=self.device) for l in group_layers], dim=1)
                       for name in ("k", "v", "g", "beta")}
            for t in range(len(stacked["k"]) - 1, spans[0][0] - 1, -1):
                k, v, b = stacked["k"][t], stacked["v"][t], stacked["beta"][t]
                gamma = stacked["g"][t].exp().clamp_min(1e-4)
                denom = (1 - b * (k*k).sum(-1)).clamp_min(1e-4)
                Y = S - b[:, None, None] * k[:, :, None] * v[:, None, :]
                S = (Y + (b / denom)[:, None, None] * k[:, :, None] *
                     (k[:, :, None] * Y).sum(-2)[:, None, :]) / gamma[:, None, None]
                norms = S.norm(dim=(-2,-1), keepdim=True)
                clamps += int(((~torch.isfinite(norms)) | (norms > 1e4 * ref)).sum())
                S = torch.nan_to_num(S, nan=0, posinf=0, neginf=0)
                S *= (1e4 * ref / S.norm(dim=(-2,-1), keepdim=True).clamp_min(1)).clamp_max(1)
            S = self.operand_replay(stacked, ix, S.unsqueeze(0))[0]
            for i, l in enumerate(group_layers):
                work.update_recurrent_state(S[i*hv:(i+1)*hv].unsqueeze(0), l)
            del stacked, S, ref
        return work, clamps

    def relative_states(self, candidate, oracle):
        return {str(l): float((candidate.get_recurrent_state(l, 0).float() -
                              oracle.get_recurrent_state(l, 0).float()).norm() /
                             oracle.get_recurrent_state(l, 0).float().norm().clamp_min(1e-12))
                for l in self.gdn}

    def adjoint_real_check(self, op1, op2, q):
        """R2.25/R2.28 on real two-trajectory operands, FP64, batched heads.

        Store only adjoint vectors, never a T-by-state-matrix history. FP32
        saturation is checked BEFORE conversion and reported as invalid.
        """
        torch = self.torch
        def valid_detail(op):
            gamma, kn = op["g"].exp(), op["k"].square().sum(-1)
            z = op["beta"] * kn
            ok = (gamma > 0) & (gamma <= 1) & (kn <= 1) & (z >= 0) & (z < 1)
            return ok, gamma, kn, z
        v1, g1, kn1, z1 = valid_detail(op1)
        v2, g2, kn2, z2 = valid_detail(op2)
        validity = v1 & v2
        gamma_ok = ((g1 > 0) & (g1 <= 1) & (g2 > 0) & (g2 <= 1)).all(0)
        kn_ok = ((kn1 <= 1) & (kn2 <= 1)).all(0)
        kn_ulp_ok = ((kn1 <= 1 + 1e-5) & (kn2 <= 1 + 1e-5)).all(0)
        z_ok = ((z1 >= 0) & (z1 < 1) & (z2 >= 0) & (z2 < 1)).all(0)
        a, b = ({k: v.double() for k, v in op.items()} for op in (op1, op2))
        n, hv, dk = a["k"].shape
        dv = a["v"].shape[-1]
        p = q.double().clone()
        ps, gammas = [], []
        cg = torch.ones(hv, device=self.device, dtype=torch.float64)
        for t in range(n-1, -1, -1):
            ps.append(p.clone()); gammas.append(cg.clone())
            k, beta, gamma = b["k"][t], b["beta"][t], b["g"][t].exp()
            p = gamma[:, None] * (p - beta[:, None] * k * (k*p).sum(-1)[:, None])
            cg *= gamma
        S1 = torch.zeros(hv, dk, dv, device=self.device, dtype=torch.float64)
        S2 = torch.zeros_like(S1)
        rhs = torch.zeros(hv, dv, device=self.device, dtype=torch.float64)
        adj = torch.zeros(hv, device=self.device, dtype=torch.float64)
        frob, directional = adj.clone(), adj.clone()
        qn = q.double().norm(dim=-1)
        def step(S, op, t):
            k, beta, gamma = op["k"][t], op["beta"][t], op["g"][t].exp()
            dec = S * gamma[:, None, None]
            return dec + k[:, :, None] * (beta[:, None] *
                   (op["v"][t] - (k[:, :, None] * dec).sum(-2)))[:, None, :]
        for t in range(n):
            next1 = step(S1, a, t)
            defect = next1 - step(S1, b, t)
            S1, S2 = next1, step(S2, b, t)
            p_t = ps[n-1-t]
            projected = (p_t[:, :, None] * defect).sum(-2)
            rhs += projected
            dn = defect.norm(dim=(-2,-1))
            adj += p_t.norm(dim=-1) * dn
            directional += projected.norm(dim=-1)
            frob += qn * gammas[n-1-t] * dn
        lhs = (q.double()[:, :, None] * (S1-S2)).sum(-2)
        actual = lhs.norm(dim=-1)
        head_valid = validity.all(0)
        slack = 1e-9 * (1 + frob)
        violations = head_valid & ((actual > adj + slack) | (adj > frob + slack) |
                                   (actual > directional + slack))
        return {"r23_valid": int(head_valid.sum()), "assumption_invalid": int((~head_valid).sum()),
                "assumption_invalid_tokens": int((~validity).sum()),
                "invalid_gamma_heads": int((~gamma_ok).sum()),
                "invalid_kn_gt_1_heads": int((~kn_ok).sum()),
                "invalid_kn_gt_1_plus_1e5_heads": int((~kn_ulp_ok).sum()),
                "invalid_z_ge_1_heads": int((~z_ok).sum()),
                "max_kn_fp32": float(max(kn1.max(), kn2.max())),
                "max_beta_fp32": float(max(op1["beta"].max(), op2["beta"].max())),
                "min_gamma_fp32": float(min(g1.min(), g2.min())),
                "prop8b_violations": int(violations.sum()),
                "clean_suffix_operands_included": True,
                "identity_max_abs_error": float((lhs-rhs).abs().max()),
                "actual_readout_error": actual.tolist(), "B_adj": adj.tolist(), "B_Frob": frob.tolist(),
                "B_Frob_over_B_adj": (frob / adj.clamp_min(1e-300)).tolist()}


def kl_from_logits(reference, method):
    import torch
    lp = torch.log_softmax(reference.double(), -1)
    lq = torch.log_softmax(method.double(), -1)
    value = float((lp.exp() * (lp-lq)).sum())
    if not math.isfinite(value):
        raise RuntimeError("NONFINITE_KL_ERROR")
    return max(0.0, value)


def score_method(decoded, gapped, compact, row, latency, tokens, calls=0, decode_ms=0):
    pred = decoded["pred"]
    valid = bool(re.fullmatch(r"[A-Za-z]+-\d{4}-[A-Za-z]+", pred)) and not decoded["truncated"]
    logits = decoded["first_logits"]
    return {"pred": pred, "EM_gold": int(valid and pred.lower() == row["v_gold"].lower()),
            "F1_gold": token_f1(pred, row["v_gold"]),
            "EM_vs_gapped": int(valid and pred.lower() == gapped["pred"].lower()),
            "EM_vs_compact": int(valid and pred.lower() == compact["pred"].lower()),
            "Stale_Attraction": int(row["v_stale"].lower() in pred.lower()),
            "KL_gapped": kl_from_logits(gapped["first_logits"], logits),
            "KL_compact": kl_from_logits(compact["first_logits"], logits),
            "Top1_Agree_gapped": int(int(logits.argmax()) == int(gapped["first_logits"].argmax())),
            "Top1_Agree_compact": int(int(logits.argmax()) == int(compact["first_logits"].argmax())),
            "edit_latency_ms": latency, "decode_ms": decode_ms,
            "replayed_layer_tokens": tokens, "clean_suffix_fa_mlp_calls": calls,
            "truncated": decoded["truncated"], "malformed": not valid}


def allocated_tensor_bytes(value):
    """Unique tensor storage payload (no allocator/object-header claim)."""
    seen = set()
    def visit(x):
        if hasattr(x, "untyped_storage"):
            storage = x.untyped_storage()
            key = (str(x.device), storage.data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(x, dict):
            return sum(visit(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return sum(visit(v) for v in x)
        return 0
    return visit(value)


def evaluate_instance(engine, row, C, shared=None, tiny=False):
    """One system cell (§4A) strictly isolated from the post-§4A mechanistic matrix (§4B)."""
    e, torch = engine, engine.torch
    T = row["T"]
    body, query = e.tensor(row["body_ids"]), e.tensor(row["query_ids"])
    pos = e.positions(T)
    plan = plan_chymera_execution(T, C, row["spans"])
    pi = torch.tensor(plan["geometry"]["pi"], device=e.device, dtype=torch.long)
    clean, gpos, Lp = body.index_select(1, pi), pi.unsqueeze(0), len(pi)
    mech_needed = row["split"] == "dev" or row["task"] == "TaskB_RAGEvict"
    if shared is None:
        # Untimed warmup is done once per model before this function.
        mono, base_ms = e.timed(lambda: e.forward(body, positions=pos).past_key_values)
        compact_cache, compact_ms = e.timed(lambda: e.counted_replay(lambda: e.forward(clean, positions=e.positions(Lp)).past_key_values))
        compact_calls = e.clean_suffix_fa_mlp_calls
        compact, compact_decode_ms = e.timed(lambda: e.greedy_decode(compact_cache, query, Lp))
        del compact_cache
        kv_cache, kv_ms = e.timed(lambda: e.evict(mono, pi))
        kv, kv_decode_ms = e.timed(lambda: e.greedy_decode(kv_cache, query, T))
        del kv_cache, mono
        shared = dict(base_ms=base_ms, compact=compact, compact_ms=compact_ms,
                      compact_decode_ms=compact_decode_ms, compact_calls=compact_calls,
                      kv=kv, kv_ms=kv_ms, kv_decode_ms=kv_decode_ms)
    # Hard isolation assertion: zero F-Oracle full-layer input buffers exist during §4A.
    assert e.f_oracle_buffer is None and "full_h" not in shared and not e.h and not e.ops, "SECTION_4A_F_ORACLE_ISOLATION_VIOLATION"
    archive, arch_ms = e.timed(lambda: e.build_archive(body, pos, C))
    kv_evict_ms = shared["kv_ms"]
    sparse, sparse_ms = e.timed(lambda: e.sparse_replay(archive, plan, clean, gpos))
    sparse_calls = e.clean_suffix_fa_mlp_calls
    gapped, gapped_decode_ms = e.timed(lambda: e.greedy_decode(sparse, query, T))
    def clm():
        w = e.evict(archive["base_cache"], pi)
        c = plan["first_non_clean_chunk"]
        for l in e.gdn:
            if c:
                e.set_state(w, l, archive["anchor_S"][c][l], archive["anchor_conv"][c][l])
            else:
                w.layers[l].recurrent_states[0].zero_()
                w.layers[l].conv_states[0].zero_()
        return w
    methods, decoded = {}, {}
    def evaluate(name, cache, ms, n_tokens):
        dec, decode_ms = e.timed(lambda: e.greedy_decode(cache, query, T))
        decoded[name] = dec
        methods[name] = score_method(dec, gapped, shared["compact"], row, ms, n_tokens, decode_ms=decode_ms)
    methods["Sparse-Anchor-SuffixReplay"] = score_method(gapped, gapped, shared["compact"], row,
                          sparse_ms, (Lp-plan["prefix_surv_len"])*e.cfg.num_hidden_layers,
                          calls=sparse_calls, decode_ms=gapped_decode_ms)
    # Oracle identity is a distribution/cache comparison even on malformed output.
    methods["Sparse-Anchor-SuffixReplay"].update(KL_gapped=0.0, EM_vs_gapped=1)
    decoded["Sparse-Anchor-SuffixReplay"] = gapped
    methods["Full-Prefill-Compact"] = score_method(shared["compact"], gapped, shared["compact"], row,
                          shared["compact_ms"], Lp*e.cfg.num_hidden_layers,
                          calls=shared["compact_calls"] if any(seg["kind"]=="clean_bmm" for seg in plan["segments"]) else 0, decode_ms=shared["compact_decode_ms"])
    methods["KV-Only-Evict"] = score_method(shared["kv"], gapped, shared["compact"], row,
                          shared["kv_ms"], 0, decode_ms=shared["kv_decode_ms"])
    decoded.update({"KV-Only-Evict": shared["kv"], "Full-Prefill-Compact": shared["compact"]})
    cache, ms = e.timed(clm)
    evaluate("CLM-NearestAnchorStaleGDN", cache, ms, 0)
    del cache
    e.jump_max_abs_err = 0.0
    e.jump_max_rel_err = 0.0
    (f_cache, seam_F, f_audit_meta), f_ms = e.timed(lambda: e.repair_F(archive, plan, body, pos, pi))
    f_calls = e.clean_suffix_fa_mlp_calls
    # Include both original dirty recovery and survivor GDN repair in F work.
    dirty_orig = sum(r["orig_hi"]-r["orig_lo"] for r in plan["chunks"] if r["status"] == "dirty")
    assert f_audit_meta["pass1_replayed_orig_tokens"] == dirty_orig, "PASS1_ORIG_TOKEN_COUNT_MISMATCH"
    evaluate("CHymERA-F", f_cache, f_ms,
             dirty_orig*e.cfg.num_hidden_layers + plan["replayed_tokens"]*len(e.gdn))
    methods["CHymERA-F"]["clean_suffix_fa_mlp_calls"] = f_calls
    methods["CHymERA-F"]["pass1_audit"] = f_audit_meta
    (fast_cache, seam), fast_ms = e.timed(lambda: e.repair_fast(archive, plan, clean, gpos, pi))
    fast_calls = e.clean_suffix_fa_mlp_calls
    direct_vs_delta_max_abs_err = float(e.jump_max_abs_err)
    direct_vs_delta_rel_err = float(e.jump_max_rel_err)
    evaluate("CHymERA-Fast", fast_cache, fast_ms, plan["replayed_tokens"]*e.cfg.num_hidden_layers)
    methods["CHymERA-Fast"]["clean_suffix_fa_mlp_calls"] = fast_calls
    q, probe_ms = e.timed(lambda: e.probe(fast_cache, query, T))
    scores, score_ms = e.timed(lambda: e.frozen_scores(archive, plan, seam, seam_F, q))
    closure_sets = [closure_for_tau(scores["eta_b_max"], tau) for tau in TAU_SWEEP]
    monotone = all(set(a).issubset(b) for a,b in zip(closure_sets, closure_sets[1:]))
    assert monotone, "NON_MONOTONE_CLOSURE"
    # Real startup endpoint check, not substitution of the reference result.
    check_key = (C, plan["first_non_clean_chunk"] == 0)
    if check_key not in e.startup_closure_checks:
        endpoint, endpoint_ms = e.timed(lambda: e.replay_upstream_blocks(e.n_blocks-1, fast_cache, archive, plan, clean, gpos))
        endpoint_diff, endpoint_rel_diff = e.cache_difference(endpoint, sparse, return_rel=True)
        assert math.isfinite(endpoint_diff) and endpoint_rel_diff <= 1e-5, (
            f"UPSTREAM_FULL_CLOSURE_MISMATCH: abs={endpoint_diff}, rel={endpoint_rel_diff}"
        )
        e.startup_closure_checks[check_key] = (endpoint_diff, endpoint_rel_diff)
        del endpoint
    endpoint_diff, endpoint_rel_diff = e.startup_closure_checks[check_key]
    closure_results = {
        -1: {**methods["CHymERA-Fast"], "unrefreshed_clean_suffix_fa_mlp_calls": 0},
        e.n_blocks-1: {**methods["Sparse-Anchor-SuffixReplay"], "unrefreshed_clean_suffix_fa_mlp_calls": 0},
    }
    required = {max(s, default=-1) for s in closure_sets}
    if row["split"] == "dev" and (C == 256 or tiny):
        required.update(range(e.n_blocks))
    closure_curve = []
    for b in sorted(required):
        if b not in closure_results:
            w, ms = e.timed(lambda b=b: e.replay_upstream_blocks(b, fast_cache, archive, plan, clean, gpos))
            closure_calls = e.clean_suffix_fa_mlp_calls
            unrefreshed_calls = e.unrefreshed_block_fa_mlp_calls
            assert unrefreshed_calls == 0, "UNREFRESHED_CLEAN_SUFFIX_CALL_VIOLATION"
            dec, dms = e.timed(lambda: e.greedy_decode(w, query, T))
            res_b = score_method(dec, gapped, shared["compact"], row,
                            ms, (Lp-plan["prefix_surv_len"])*4*(b+1), calls=closure_calls, decode_ms=dms)
            res_b["unrefreshed_clean_suffix_fa_mlp_calls"] = unrefreshed_calls
            closure_results[b] = res_b
            del w
        closure_curve.append({"b_star": b, "eta_b_max": scores["eta_b_max"][b] if b >= 0 else None,
                              "KL_gapped_after_closure_b": closure_results[b]["KL_gapped"]})
    seam_score_total_ms = probe_ms + score_ms
    for tau, closure in zip(TAU_SWEEP, closure_sets):
        b = max(closure, default=-1)
        metric = dict(closure_results[b])
        if tau == 0.0:
            metric.update(
                edit_latency_ms=sparse_ms,
                seam_score_ms=0.0,
                closure_replay_ms=sparse_ms,
                replayed_layer_tokens=methods["Sparse-Anchor-SuffixReplay"]["replayed_layer_tokens"],
                unrefreshed_clean_suffix_fa_mlp_calls=0,
            )
        elif math.isinf(tau):
            assert b == -1, "TAU_INF_MUST_SELECT_EMPTY_CLOSURE"
            metric.update(
                edit_latency_ms=fast_ms,
                seam_score_ms=0.0,
                closure_replay_ms=0.0,
                replayed_layer_tokens=methods["CHymERA-Fast"]["replayed_layer_tokens"],
                unrefreshed_clean_suffix_fa_mlp_calls=0,
            )
        else:
            extra = 0.0 if b == -1 else (sparse_ms if b == e.n_blocks-1 else closure_results[b]["edit_latency_ms"])
            metric.update(
                edit_latency_ms=fast_ms + seam_score_total_ms + extra,
                seam_score_ms=seam_score_total_ms,
                closure_replay_ms=extra,
                replayed_layer_tokens=methods["CHymERA-Fast"]["replayed_layer_tokens"] +
                                      (0 if b == -1 else (Lp-plan["prefix_surv_len"])*4*(b+1)),
                unrefreshed_clean_suffix_fa_mlp_calls=closure_results[b]["unrefreshed_clean_suffix_fa_mlp_calls"],
            )
        metric.update(closure_blocks=closure, b_star=b,
                      extra_replayed_layer_tokens=(Lp-plan["prefix_surv_len"])*4*(b+1) if b>=0 else 0)
        methods[tau_name(tau)] = metric
    assert methods[tau_name(0.0)]["KL_gapped"] < 1e-6
    assert e.f_oracle_buffer is None, "POST_4A_F_ORACLE_ISOLATION_VIOLATION"
    del sparse

    # §4B Dedicated Mechanistic Ablation Matrix: allocated strictly AFTER §4A completes and freed immediately.
    # Captured layer inputs/operands run at chunk_stride=C matching build_archive(body, pos, C) and are offloaded per-layer to CPU RAM so 4B/9B never OOM GPU VRAM.
    mech = None
    if mech_needed:
        if e.device.type == "cuda":
            torch.cuda.empty_cache()
        mono_cap, h_cap, ops_cap = e.capture(body, pos, offload_cpu=True, chunk_stride=C)
        del mono_cap
        if e.device.type == "cuda":
            torch.cuda.empty_cache()
        oracle, oracle_ms = e.timed(lambda: e.f_oracle(archive, pi, h_cap))
        nohalo, nohalo_ms = e.timed(lambda: e.nohalo(archive, plan, pi, ops_cap))
        ablations = {}
        def ablate(name, w, ms):
            dec, dms = e.timed(lambda: e.greedy_decode(w, query, T))
            decoded[name] = dec
            ablations[name] = score_method(dec, gapped, shared["compact"], row, ms,
                                           Lp*len(e.gdn), decode_ms=dms)
        ablate("F-Oracle-FullLayerInput", oracle, oracle_ms)
        ablate("NoHalo-ChunkCompose", nohalo, nohalo_ms)
        err_F, err_fast = e.relative_states(f_cache, oracle), e.relative_states(fast_cache, oracle)
        err_nohalo, err_naive = e.relative_states(nohalo, oracle), {}
        del nohalo
        if len(row["spans"]) > 1:
            naive_ix = e.naive_indices(row["spans"], T)
            # Fixed-input per-layer F diagnostic uses EXACT original h at naive indices.
            naive, naive_ms = e.timed(lambda: e.f_oracle(archive, naive_ix, h_cap))
            # Restore survivor FA KV so this is an F-state ablation.
            for l in e.fa:
                naive.layers[l] = copy.deepcopy(oracle.layers[l])
            err_naive = e.relative_states(naive, oracle)
            ablate("Naive-PerSpanHalo", naive, naive_ms)
            del naive
            # Also log the separate actual multilayer naive end-to-end decoding.
            naive_N, naive_N_ms = e.timed(lambda: e.forward(body.index_select(1, naive_ix),
                                                    positions=naive_ix.unsqueeze(0)).past_key_values)
            dec, dms = e.timed(lambda: e.greedy_decode(naive_N, query, T))
            ablations["Naive-PerSpanHalo"]["end_to_end"] = score_method(dec, gapped, shared["compact"], row,
                                                 naive_N_ms, len(naive_ix)*e.cfg.num_hidden_layers, decode_ms=dms)
            del naive_N
        del h_cap
        clamp_events = None
        if row["split"] == "dev" and (C == 256 or tiny):
            (inverse, clamp_events), inverse_ms = e.timed(lambda: e.terminal_inverse(archive, pi, row["spans"], ops_cap))
            ablate("Terminal-Inverse", inverse, inverse_ms)
            del inverse
        # Context after query ends in newline (Instruct) or space (Base).
        adapter = FastTokenizerAdapter(e.tokenizer)
        def first_id(value):
            # The fixed query IDs have already been fed. Its trailing whitespace
            # is resident, so a continuation starts with encode(value)[0].
            plain = adapter.encode_plain(value)[0]
            return plain, {"plain": plain, "leading_space": adapter.encode_plain(" " + value)[0],
                           "selected": plain, "query_tail": adapter.decode_plain(row["query_ids"])[-16:]}
        gold_id, gold_variants = first_id(row["v_gold"])
        stale_id, stale_variants = first_id(row["v_stale"])
        margins = {name: float(dec["first_logits"][gold_id]-dec["first_logits"][stale_id]) for name, dec in decoded.items()}
        real_checks = {}
        if row["split"] == "dev" and (C == 256 or tiny):
            # Distinct original-survivor and gapped N trajectories, both real weights.
            gapped_full, _, edited_ops = e.capture(clean, gpos, offload_cpu=True, include_h=False)
            del gapped_full
            pi_cpu = pi.detach().cpu()
            for l in e.gdn:
                op1 = {k: v.index_select(0, pi_cpu).to(device=e.device) for k, v in ops_cap[l].items()}
                op2 = {k: v.to(device=e.device) for k, v in edited_ops[l].items()}
                real_checks[str(l)] = e.adjoint_real_check(op1, op2, q[l])
                del op1, op2
            del edited_ops
        pure_suffix = {}
        if seam:
            suffix = plan["segments"][-1]["chunks"]
            for l in e.gdn:
                p, gamma_product = q[l].double().unsqueeze(-1), torch.ones(q[l].shape[0], device=e.device, dtype=torch.float64)
                propagated = seam[l].double()
                for c in suffix:
                    propagated = torch.bmm(archive["PH"][c][l][0].double(), propagated)
                for c in reversed(suffix):
                    p = torch.bmm(archive["PH"][c][l][0].double().transpose(-1,-2), p)
                lo = archive["chunks"][suffix[0]][0]
                op = {k: v[lo:].to(device=e.device) for k, v in ops_cap[l].items()}
                gamma = op["g"].exp()
                kn = op["k"].square().sum(-1)
                bn = op["beta"] * kn
                gamma_ok = ((gamma > 0) & (gamma <= 1)).all(0)
                kn_ok = (kn <= 1).all(0)
                kn_ulp_ok = (kn <= 1 + 1e-5).all(0)
                z_ok = ((bn >= 0) & (bn < 1)).all(0)
                valid = ((gamma>0)&(gamma<=1)&(kn<=1)&(bn>=0)&(bn<1)).all(0)
                max_kn_fp32 = float(kn.max())
                max_beta_fp32 = float(op["beta"].max())
                min_gamma_fp32 = float(gamma.min())
                gamma_product *= op["g"].double().sum(0).exp()
                del op
                actual = (q[l].double()[:,:,None]*propagated).sum(-2).norm(dim=-1)
                adj = p.squeeze(-1).norm(dim=-1)*seam[l].double().norm(dim=(-2,-1))
                frob = q[l].double().norm(dim=-1)*gamma_product*seam[l].double().norm(dim=(-2,-1))
                tol = 1e-6*(1+frob)
                pure_suffix[str(l)] = {"r23_valid": int(valid.sum()), "assumption_invalid": int((~valid).sum()),
                    "invalid_gamma_heads": int((~gamma_ok).sum()),
                    "invalid_kn_gt_1_heads": int((~kn_ok).sum()),
                    "invalid_kn_gt_1_plus_1e5_heads": int((~kn_ulp_ok).sum()),
                    "invalid_z_ge_1_heads": int((~z_ok).sum()),
                    "max_kn_fp32": max_kn_fp32, "max_beta_fp32": max_beta_fp32, "min_gamma_fp32": min_gamma_fp32,
                    "prop8b_violations": int((valid & ((actual>adj+tol)|(adj>frob+tol))).sum()),
                    "actual_readout_error": actual.tolist(), "B_adj": adj.tolist(), "B_Frob": frob.tolist(),
                    "B_Frob_over_B_adj": (frob/adj.clamp_min(1e-300)).tolist()}
        mech = {"model": row.get("model"), "instance_id": row["instance_id"], "task": row["task"],
                "sub_regime": row["sub_regime"], "split": row["split"], "T": T, "C": C,
                "f_oracle_isolated_from_4A": True,
                "rel_err_chymera_F_by_layer": err_F, "rel_err_chymera_Fast_by_layer": err_fast,
                "rel_err_nohalo_by_layer": err_nohalo, "rel_err_naive_halo_by_layer": err_naive,
                "norm_delta_S_F_vs_unedited_by_layer": {str(l): float((oracle.get_recurrent_state(l,0)-archive["base_cache"].get_recurrent_state(l,0)).norm()) for l in e.gdn},
                "conv_halo_residual_by_layer": err_nohalo,
                "conv_state_rel_err_F_by_layer": {str(l): float((f_cache.get_conv_state(l,0).float()-oracle.get_conv_state(l,0).float()).norm()/oracle.get_conv_state(l,0).float().norm().clamp_min(1e-12)) for l in e.gdn},
                "first_token_ids": {"gold": gold_variants, "stale": stale_variants}, "margins": margins,
                "margins_leading_space_variant": {name: float(dec["first_logits"][gold_variants["leading_space"]] - dec["first_logits"][stale_variants["leading_space"]]) for name, dec in decoded.items()},
                "Ghost_GDN_pull": margins["CHymERA-F"]-margins["KV-Only-Evict"],
                "Conv_Halo_pull": margins["CHymERA-F"]-margins["NoHalo-ChunkCompose"],
                "Suffix_StaleKV_pull": margins["Sparse-Anchor-SuffixReplay"]-margins["CHymERA-F"],
                "real_weight_adjoint": real_checks, "pure_suffix_adjoint": pure_suffix,
                "blockwise_closure_trajectory": closure_curve, "terminal_inv_clamps": clamp_events,
                "ablations": ablations}
        del oracle, ops_cap
        assert e.f_oracle_buffer is None
    allocated = allocated_tensor_bytes({k:archive[k] for k in ("anchor_S", "anchor_conv", "PH", "Gamma")})
    base = archive["base_cache"]
    kv_bytes = allocated_tensor_bytes([(base.layers[l].keys, base.layers[l].values) for l in e.fa])
    cache_bytes = allocated_tensor_bytes([(base.get_recurrent_state(l,0), base.get_conv_state(l,0)) for l in e.gdn]) + kv_bytes
    result = {k:row[k] for k in ("instance_id", "task", "sub_regime", "split", "T", "v_gold", "v_stale")}
    result.update(model=row.get("model"), C=C, R=len(plan["geometry"]["canonical_spans"]), L_prime=Lp,
                  methods=methods, **scores, eta_gate_source="fast_seam",
                  eta_gate_source_note="CHymERA-Fast pre-refresh seam drives runtime gate; F-seam adjoint reported separately as diagnostic F archive certificate",
                  eta_adj_max=max(scores["eta_b_max"], default=0),
                  eta_frob_max=max(scores["eta_b_frob"], default=0), closure_monotone=monotone,
                  direct_vs_delta_max_abs_err=direct_vs_delta_max_abs_err,
                  direct_vs_delta_rel_err=direct_vs_delta_rel_err,
                  full_closure_cache_max_abs_diff=endpoint_diff,
                  full_closure_cache_max_rel_diff=endpoint_rel_diff,
                  full_closure_check_scope="startup per stride and empty/nonempty anchor",
                  prefill_base_ms=shared["base_ms"], prefill_chymera_ms=arch_ms,
                  prefill_wy_fold_ms=max(0, arch_ms-shared["base_ms"]), kv_evict_ms=kv_evict_ms,
                  query_probe_ms=probe_ms, gate_score_ms=score_ms,
                  seam_score_ms=seam_score_total_ms,
                  extra_archive_bytes=allocated, extra_archive_mib=allocated/2**20,
                  actual_dynamic_cache_bytes=cache_bytes, actual_full_attn_kv_bytes=kv_bytes,
                  extra_archive_over_kv_ratio=allocated/max(kv_bytes,1),
                  replayed_dirty_tokens=plan["replayed_tokens"],
                  dirty_token_bound=min(Lp, len(plan["geometry"]["canonical_spans"])*(2*C+CONV_W-3)))
    del f_cache, fast_cache
    return result, mech, shared, archive

def benchmark_task_c_multiedit(engine, row, C, base_archive, prefill_chymera_ms, prefill_wy_fold_ms):
    """Real independent branches and online cumulative edits, with symmetric prefill accounting.

    Rebuild is deliberately conservative: full surviving archive, including all
    invalidated summaries and downstream anchors. It is separately charged and
    never called an incremental/local rebuild.
    """
    e, torch = engine, engine.torch
    s, hi = row["P_pin"], row["W_keep"]
    edges = np.linspace(s, hi, 5, dtype=np.int64)
    subspans = [(int(edges[i]), int(edges[i+1])-1) for i in range(4)]
    assert all(a <= b for a,b in subspans)
    body, query, positions = e.tensor(row["body_ids"]), e.tensor(row["query_ids"]), e.positions(row["T"])
    anchor_only_archive, prefill_anchor_ms = e.timed(lambda: e.build_anchor_only_archive(body, positions, C))
    names = ("CHymERA-Fast", "CHymERA-F", "Sparse-Anchor-SuffixReplay", "Full-Prefill-Compact")
    def edit(name, archive, current, pos, span, decode_branch=False):
        plan = plan_chymera_execution(current.shape[1], C, [span])
        ix = torch.tensor(plan["geometry"]["pi"], device=e.device, dtype=torch.long)
        clean, gpos = current.index_select(1, ix), pos.index_select(1, ix)
        if name == "CHymERA-Fast":
            fn = lambda: e.repair_fast(archive, plan, clean, gpos, ix)
        elif name == "CHymERA-F":
            fn = lambda: e.repair_F(archive, plan, current, pos, ix)
        elif name == "Sparse-Anchor-SuffixReplay":
            fn = lambda: e.sparse_replay(archive, plan, clean, gpos)
        else:
            fn = lambda: e.forward(clean, positions=e.positions(clean.shape[1])).past_key_values
        value, ms = e.timed(fn)
        em_gold = None
        if decode_branch:
            cache_obj = value[0] if isinstance(value, tuple) else value
            start_pos = clean.shape[1] if name == "Full-Prefill-Compact" else row["T"]
            dec = e.greedy_decode(cache_obj, query, start_pos)
            valid = bool(re.fullmatch(r"[A-Za-z]+-\d{4}-[A-Za-z]+", dec["pred"])) and not dec["truncated"]
            em_gold = int(valid and dec["pred"].lower() == row["v_gold"].lower())
        del value
        return clean, gpos, ix, ms, em_gold
    # Four DISTINCT single-edit masks on immutable archive, measured once each (pure edit_ms + branch EM_gold).
    branch_edits = {name: [] for name in names}
    branch_ems = {name: [] for name in names}
    for span in subspans:
        for name in names:
            arch_for_branch = anchor_only_archive if name == "Sparse-Anchor-SuffixReplay" else base_archive
            _, _, _, ms, em_gold = edit(name, arch_for_branch, body, positions, span, decode_branch=True)
            branch_edits[name].append(ms)
            branch_ems[name].append(em_gold)
    protocols = []
    for K in (1,2,4):
        for name in names:
            wy_overhead = prefill_wy_fold_ms if name.startswith("CHymERA") else 0.0
            sym_prefill = (prefill_chymera_ms if name.startswith("CHymERA")
                           else prefill_anchor_ms if name == "Sparse-Anchor-SuffixReplay"
                           else 0.0)
            measurements = branch_edits[name][:K]
            ems_k = branch_ems[name][:K]
            protocols.append({"protocol": "Branching-Counterfactual", "K": K, "method": name,
                "edit_ms": measurements, "sum_edit_ms": sum(measurements), "rebuild_ms": [],
                "archive_rebuilds": 0, "prefill_wy_fold_ms": wy_overhead,
                "symmetric_prefill_ms": sym_prefill,
                "symmetric_total_ms": sym_prefill + sum(measurements),
                "net_amortized_ms": wy_overhead + sum(measurements),
                "total_ms": sym_prefill + sum(measurements),
                "branch_EM_gold": ems_k,
                "mean_branch_EM_gold": float(np.mean(ems_k)),
                "distinct_masks": subspans[:K]})
            current, curpos = body, positions
            archive = anchor_only_archive if name == "Sparse-Anchor-SuffixReplay" else base_archive
            original_ix = torch.arange(row["T"], device=e.device)
            edits, rebuilds, rebuild_lengths = [], [], []
            for k in range(K):
                a,b = subspans[k]
                local = torch.nonzero((original_ix >= a)&(original_ix <= b)).flatten()
                assert len(local) == b-a+1, "CUMULATIVE_MASK_COORDINATE_ERROR"
                span = (int(local[0]), int(local[-1]))
                current, curpos, ix, ms, _ = edit(name, archive, current, curpos, span, decode_branch=False)
                original_ix = original_ix.index_select(0, ix)
                edits.append(ms)
                if k+1 < K and name != "Full-Prefill-Compact":
                    # Rebuild surviving archive (anchor-only for SparseAnchor, WY+anchors for CHymERA).
                    if name == "Sparse-Anchor-SuffixReplay":
                        archive, rebuild_ms = e.timed(lambda: e.build_anchor_only_archive(current, curpos, C))
                    else:
                        archive, rebuild_ms = e.timed(lambda: e.build_archive(current, curpos, C))
                    rebuilds.append(rebuild_ms); rebuild_lengths.append(current.shape[1])
            protocols.append({"protocol": "Cumulative-Rebuild", "K": K, "method": name,
                "edit_ms": edits, "sum_edit_ms": sum(edits), "rebuild_ms": rebuilds,
                "sum_rebuild_ms": sum(rebuilds), "archive_rebuilds": len(rebuilds),
                "rebuild_policy": "full-surviving-archive (conservative)", "rebuild_T": rebuild_lengths,
                "prefill_wy_fold_ms": wy_overhead,
                "symmetric_prefill_ms": sym_prefill,
                "total_cum_ms": sym_prefill + sum(edits) + sum(rebuilds),
                "net_amortized_cum_ms": wy_overhead + sum(edits) + sum(rebuilds),
                "final_original_survivor_ids": original_ix.tolist()})
    del anchor_only_archive
    break_even = {}
    for name in ("CHymERA-F", "CHymERA-Fast"):
        break_even[name] = {}
        for baseline in ("Sparse-Anchor-SuffixReplay", "Full-Prefill-Compact"):
            base_prefill = prefill_anchor_ms if baseline == "Sparse-Anchor-SuffixReplay" else 0.0
            extra_prefill_sym = max(0.0, prefill_chymera_ms - base_prefill)
            savings = float(np.mean(branch_edits[baseline])-np.mean(branch_edits[name]))
            kstar_sym = max(1, math.ceil(extra_prefill_sym/savings)) if savings > 0 else math.inf
            kstar_wy = max(1, math.ceil(prefill_wy_fold_ms/savings)) if savings > 0 else math.inf
            observed = [K for K in (1,2,4)
                        if prefill_chymera_ms + sum(branch_edits[name][:K]) < base_prefill + sum(branch_edits[baseline][:K])]
            break_even[name][baseline] = {
                "K_star": kstar_sym,
                "K_star_wy_fold_only": kstar_wy,
                "extra_prefill_ms_vs_baseline": extra_prefill_sym,
                "savings_ms_per_edit": savings,
                "definition": "ceil(max(0, prefill_chymera_ms - prefill_baseline_ms) / measured mean branch savings)",
                "observed_break_even_K": min(observed) if observed else None,
                "extrapolated_beyond_four": kstar_sym > 4,
            }
    return {"instance_id": row["instance_id"], "split": row["split"], "T": row["T"], "C": C,
            "prefill_chymera_ms": prefill_chymera_ms, "prefill_anchor_ms": prefill_anchor_ms,
            "prefill_wy_fold_ms": prefill_wy_fold_ms,
            "subspans": subspans, "protocols": protocols, "break_even": break_even}


def bootstrap_statistics(diffs, alpha=0.05, seed=SEED):
    arr = np.asarray(diffs, dtype=np.float64)
    if not len(arr):
        return {"mean": None, "ci95": None, "p_one_sided": None, "n": 0, "seed": seed}
    rng = np.random.default_rng(seed)
    boot = arr[rng.integers(0, len(arr), size=(2000, len(arr)))].mean(1)
    return {"mean": float(arr.mean()), "ci95": np.quantile(boot, [alpha/2,1-alpha/2]).tolist(),
            "p_one_sided": float((1+np.sum(boot <= 0))/2001), "n": len(arr), "seed": seed}


def aggregate_methods(rows):
    if not rows:
        return {}
    return {name: {metric: float(np.mean([r["methods"][name][metric] for r in rows]))
                   for metric in ("EM_gold", "EM_vs_gapped", "EM_vs_compact", "F1_gold", "Stale_Attraction",
                                  "KL_gapped", "KL_compact", "edit_latency_ms", "decode_ms", "replayed_layer_tokens")}
            for name in rows[0]["methods"]}


PRIMARY_INSTRUCT_SCALES = ("0.8B", "2B", "4B", "9B")
STALE_BASELINES = ("KV-Only-Evict", "CLM-NearestAnchorStaleGDN")


def summarize_model(tag, rows, mechanistic, multiedit, tiny=False, oom_events=None, truncated=False):
    oom_events = list(oom_events or [])
    oom_count_total = len(oom_events)
    is_primary_instruct = (tag in PRIMARY_INSTRUCT_SCALES) and (not tiny)
    scale_idx = PRIMARY_INSTRUCT_SCALES.index(tag) if tag in PRIMARY_INSTRUCT_SCALES else 0
    model_status = (
        "TRUNCATED_INCOMPLETE" if truncated
        else "COMPLETE_WITH_OOM" if oom_count_total > 0
        else "COMPLETE"
    )
    summary = {"version": "R2.1", "model": tag,
               "status": model_status,
               "truncated": bool(truncated),
               "oom_count": oom_count_total, "oom_events": oom_events,
               "tiny_arch_dryrun": tiny,
               "n_instances": len({r["instance_id"] for r in rows}), "n_system_cells": len(rows),
               "n_mechanistic_instances": len({r["instance_id"] for r in mechanistic}), "by_C": {}}
    strides_present = sorted({r["C"] for r in rows} | {e["C"] for e in oom_events if "C" in e})
    if is_primary_instruct and 256 not in strides_present:
        strides_present = sorted(set(strides_present) | {256})
    expected_methods_c = {"CHymERA-Fast", "CHymERA-F", "Sparse-Anchor-SuffixReplay", "Full-Prefill-Compact"}
    expected_protocols_c = {"Branching-Counterfactual", "Cumulative-Rebuild"}

    def _multiedit_complete_for_stride(c_target):
        if truncated or any(e.get("C") == c_target for e in oom_events):
            return False
        mc_c = [m for m in multiedit if m["C"] == c_target and m.get("split") == "heldout_test" and m["T"] == 4096]
        if len(mc_c) != 6 or len({m["instance_id"] for m in mc_c}) != 6:
            return False
        return all(
            len(e_m["protocols"]) == 24
            and {p["K"] for p in e_m["protocols"]} == {1, 2, 4}
            and {p["protocol"] for p in e_m["protocols"]} == expected_protocols_c
            and {p["method"] for p in e_m["protocols"]} == expected_methods_c
            and all(
                p["archive_rebuilds"] == (p["K"] - 1 if p["protocol"] == "Cumulative-Rebuild" and p["method"] != "Full-Prefill-Compact" else 0)
                and math.isfinite(p.get("symmetric_total_ms", p.get("total_cum_ms", math.nan)))
                and p.get("symmetric_total_ms", p.get("total_cum_ms", -1.0)) >= 0.0
                for p in e_m["protocols"]
            )
            for e_m in mc_c
        )

    for C in strides_present:
        rs = sorted([r for r in rows if r["C"] == C], key=lambda x: x["instance_id"])
        oom_c = [e for e in oom_events if e.get("C") == C]
        oom_c_count = len(oom_c)
        is_primary_gate_stride = (C == 256)
        gate_scope = "PRIMARY_PREREGISTERED_C256" if is_primary_gate_stride else "REPORTING_ONLY_NON_SELECTABLE"
        dev = sorted([r for r in rs if r["split"] == "dev"], key=lambda x: x["instance_id"])
        test = sorted([r for r in rs if r["split"] == "heldout_test"], key=lambda x: x["instance_id"])
        full_sample = (not truncated and len(test) == 54 and len(dev) == 18 and oom_c_count == 0 and is_primary_instruct)
        comparisons = {}
        for m_group_idx, name in enumerate(("CHymERA-Fast", "CHymERA-F")):
            for b_idx, baseline in enumerate(STALE_BASELINES):
                k_idx = 8 * m_group_idx + 2 * scale_idx + b_idx
                key = name + " vs " + baseline
                diffs = {"Delta_KL_gapped": [r["methods"][baseline]["KL_gapped"]-r["methods"][name]["KL_gapped"] for r in test],
                         "Delta_EM_vs_gapped": [r["methods"][name]["EM_vs_gapped"]-r["methods"][baseline]["EM_vs_gapped"] for r in test],
                         "Delta_EM_gold": [r["methods"][name]["EM_gold"]-r["methods"][baseline]["EM_gold"] for r in test]}
                med_method = float(np.median([r["methods"][name]["KL_gapped"] for r in test])) if test else None
                med_base = float(np.median([r["methods"][baseline]["KL_gapped"] for r in test])) if test else None
                ratio = med_base/max(med_method,1e-300) if test else None
                perm_p = sign_flip_permutation_p_one_sided(diffs["Delta_KL_gapped"], n_perm=2000, seed=SEED + k_idx)
                stats = {
                    "Delta_KL_gapped": bootstrap_statistics(diffs["Delta_KL_gapped"], seed=SEED + 100 + k_idx),
                    "Delta_EM_vs_gapped": bootstrap_statistics(diffs["Delta_EM_vs_gapped"], seed=SEED + 100 + k_idx),
                    "Delta_EM_gold": bootstrap_statistics(diffs["Delta_EM_gold"], seed=SEED + 100 + k_idx),
                }
                local_effect_pass = bool(
                    full_sample and med_base is not None and med_method is not None
                    and med_base > 0.0 and med_method <= (med_base / 3.0)
                )
                comparisons[key] = {
                    "k_index_M16": k_idx,
                    "permutation_seed": SEED + k_idx,
                    "ci_bootstrap_seed": SEED + 100 + k_idx,
                    "sign_flip_perm_p_one_sided": perm_p,
                    "paired_diffs": diffs,
                    "descriptive_bootstrap_ci_2000": stats,
                    "median_KL_method": med_method,
                    "median_KL_baseline": med_base,
                    "median_KL_reduction_ratio": ratio,
                    "local_pass": local_effect_pass,
                }
        # G2: All n=18 heldout_test Task B instances (6 len1 + 6 len2 + 6 spill) for F < 1e-3 & NoHalo > 10x,
        #     plus exact n=12 short-island subset (6 len1 + 6 len2) for Naive-PerSpanHalo > 10x at C=256.
        ms_taskb_heldout = [m for m in mechanistic if m["C"] == C and m.get("split") == "heldout_test"
                            and m.get("task") == "TaskB_RAGEvict"]
        ms_short_heldout = [m for m in ms_taskb_heldout if m["sub_regime"] in ("short_island_len1", "short_island_len2")]
        ms_spill_heldout = [m for m in ms_taskb_heldout if m["sub_regime"] == "chunk_boundary_halo_spill"]
        ms_dev = [m for m in mechanistic if m["C"] == C and m.get("split") == "dev"
                  and m["sub_regime"] in ("short_island_len1", "short_island_len2")]
        n_heldout_l1 = sum(1 for m in ms_short_heldout if m["sub_regime"] == "short_island_len1")
        n_heldout_l2 = sum(1 for m in ms_short_heldout if m["sub_regime"] == "short_island_len2")
        n_heldout_spill = len(ms_spill_heldout)
        err_on = lambda subset, field: max((v for m in subset for v in m[field].values()), default=None)
        ef = err_on(ms_taskb_heldout, "rel_err_chymera_F_by_layer")
        en = err_on(ms_taskb_heldout, "rel_err_nohalo_by_layer")
        eh = err_on(ms_short_heldout, "rel_err_naive_halo_by_layer")
        ef_dev, en_dev, eh_dev = (err_on(ms_dev, f) for f in ("rel_err_chymera_F_by_layer", "rel_err_nohalo_by_layer", "rel_err_naive_halo_by_layer"))
        isolated_ok = bool(ms_taskb_heldout) and all(m.get("f_oracle_isolated_from_4A", False) for m in ms_taskb_heldout) and all(
            r["methods"]["CHymERA-F"].get("pass1_audit", {}).get("f_oracle_buffer_was_none", False) for r in rs
        )
        complete_taskb = (
            len(ms_taskb_heldout) == 18 and len(ms_short_heldout) == 12
            and n_heldout_l1 == 6 and n_heldout_l2 == 6 and n_heldout_spill == 6 and is_primary_instruct
        )
        g2_numeric = bool(
            ms_taskb_heldout and ms_short_heldout and ef is not None and en is not None and eh is not None
            and ef < 1e-3 and en > 10.0 * max(ef, 1e-12) and eh > 10.0 * max(ef, 1e-12) and isolated_ok
        )
        if is_primary_gate_stride and is_primary_instruct:
            g2_pass = bool(full_sample and oom_c_count == 0 and complete_taskb and g2_numeric)
        else:
            g2_pass = None

        # G3: (i) zero clean-suffix FA/MLP calls across CHymERA-F, CHymERA-Fast, and unrefreshed blocks > b* of Certified(tau);
        #     (ii) exact numeric completeness across K in {1,2,4}, both protocols, all 4 methods;
        #     (iii) symmetric K=4 Branching-Counterfactual total_ms & EM_gold check on n=6 heldout_test T=4096 Task C instances at C=256.
        zero_calls = bool(rs) and all(
            r["methods"]["CHymERA-F"]["clean_suffix_fa_mlp_calls"] == 0
            and r["methods"]["CHymERA-Fast"]["clean_suffix_fa_mlp_calls"] == 0
            and all(r["methods"][tau_name(tau)].get("unrefreshed_clean_suffix_fa_mlp_calls", 0) == 0 for tau in TAU_SWEEP)
            for r in rs
        )
        mc_all_c = sorted([m for m in multiedit if m["C"] == C], key=lambda x: x["instance_id"])
        mc_heldout_k4 = [m for m in mc_all_c if m.get("split") == "heldout_test" and m["T"] == 4096]
        k4_sym_cell = {"n_heldout_TaskC_T4096": len(mc_heldout_k4)}
        if mc_heldout_k4:
            def proto_k4(entry, method_name):
                return next(p for p in entry["protocols"]
                            if p["protocol"] == "Branching-Counterfactual" and p["K"] == 4 and p["method"] == method_name)
            mean_tot_fast = float(np.mean([proto_k4(e_m, "CHymERA-Fast")["symmetric_total_ms"] for e_m in mc_heldout_k4]))
            mean_tot_f = float(np.mean([proto_k4(e_m, "CHymERA-F")["symmetric_total_ms"] for e_m in mc_heldout_k4]))
            mean_tot_sparse = float(np.mean([proto_k4(e_m, "Sparse-Anchor-SuffixReplay")["symmetric_total_ms"] for e_m in mc_heldout_k4]))
            mean_tot_compact = float(np.mean([proto_k4(e_m, "Full-Prefill-Compact")["symmetric_total_ms"] for e_m in mc_heldout_k4]))
            em_sparse_k4 = float(np.mean([proto_k4(e_m, "Sparse-Anchor-SuffixReplay")["mean_branch_EM_gold"] for e_m in mc_heldout_k4]))
            em_fast_k4 = float(np.mean([proto_k4(e_m, "CHymERA-Fast")["mean_branch_EM_gold"] for e_m in mc_heldout_k4]))
            k4_sym_cell.update(
                mean_total_ms_Fast_K4=mean_tot_fast,
                mean_total_ms_F_K4=mean_tot_f,
                mean_total_ms_SparseAnchor_K4=mean_tot_sparse,
                mean_total_ms_Compact_K4=mean_tot_compact,
                speedup_SparseAnchor_over_Fast_K4=mean_tot_sparse / max(mean_tot_fast, 1e-12),
                speedup_Compact_over_Fast_K4=mean_tot_compact / max(mean_tot_fast, 1e-12),
                mean_EM_gold_SparseAnchor_K4=em_sparse_k4,
                mean_EM_gold_Fast_K4=em_fast_k4,
                EM_gold_retention_Fast_vs_SparseAnchor_K4=(em_fast_k4 / em_sparse_k4) if em_sparse_k4 > 0 else 0.0,
            )
        speed_rows = [r for r in test if r["T"] == 4096 and r["R"] == 1]
        single_edit_diag = {}
        if speed_rows:
            fast_1 = float(np.mean([r["methods"]["CHymERA-Fast"]["edit_latency_ms"] for r in speed_rows]))
            for baseline in ("Sparse-Anchor-SuffixReplay", "Full-Prefill-Compact"):
                single_edit_diag[baseline] = float(np.mean([r["methods"][baseline]["edit_latency_ms"] for r in speed_rows]) / max(fast_1, 1e-12))
            em_ref_1 = float(np.mean([r["methods"]["Sparse-Anchor-SuffixReplay"]["EM_gold"] for r in speed_rows]))
            em_fast_1 = float(np.mean([r["methods"]["CHymERA-Fast"]["EM_gold"] for r in speed_rows]))
            single_edit_diag["EM_gold_retention"] = em_fast_1 / em_ref_1 if em_ref_1 > 0 else None
        if is_primary_gate_stride and is_primary_instruct:
            multiedit_numeric_ok = _multiedit_complete_for_stride(256) and _multiedit_complete_for_stride(1024)
            g3_pass = bool(
                full_sample and oom_c_count == 0 and zero_calls and multiedit_numeric_ok
                and k4_sym_cell.get("speedup_SparseAnchor_over_Fast_K4", 0.0) > 1.2
                and k4_sym_cell.get("speedup_Compact_over_Fast_K4", 0.0) > 1.5
                and k4_sym_cell.get("mean_EM_gold_SparseAnchor_K4", 0.0) >= 0.25
                and k4_sym_cell.get("mean_EM_gold_Fast_K4", 0.0) >= 0.90 * k4_sym_cell.get("mean_EM_gold_SparseAnchor_K4", 0.0)
            )
        else:
            multiedit_numeric_ok = _multiedit_complete_for_stride(C)
            g3_pass = None

        # G4: Strict 5-clause AND at C=256 (0 < tau_dev_star < inf required; Monte Carlo permutation p-value B=2000 plus-one correction on Spearman rho).
        tau_star = next((tau for tau in TAU_SWEEP if dev and np.mean([r["methods"][tau_name(tau)]["KL_gapped"] for r in dev]) <= 0.02), 0.0)
        calibrated = {"tau_dev_star": tau_star, "n_dev": len(dev), "n_heldout_test": len(test)}
        if test:
            calibrated.update(
                mean_KL_gapped=float(np.mean([r["methods"][tau_name(tau_star)]["KL_gapped"] for r in test])),
                mean_replayed_layer_tokens=float(np.mean([r["methods"][tau_name(tau_star)]["replayed_layer_tokens"] for r in test])),
                sparse_mean_replayed_layer_tokens=float(np.mean([r["methods"]["Sparse-Anchor-SuffixReplay"]["replayed_layer_tokens"] for r in test])),
            )
        mech_c = [m for m in mechanistic if m["C"] == C]
        dev_mech = [m for m in mech_c if m.get("split") == "dev"]
        all_checks = [check for m in mech_c for field in ("real_weight_adjoint", "pure_suffix_adjoint") for check in m[field].values()]
        valid_count = sum(x["r23_valid"] for x in all_checks)
        assumption_invalid = sum(x["assumption_invalid"] for x in all_checks)
        violations = sum(x["prop8b_violations"] for x in all_checks)
        nondegenerate = [r for r in test if r["methods"]["CHymERA-Fast"]["KL_gapped"] > 1e-3]
        eta_nd = [r["eta_adj_max"] for r in nondegenerate]
        kl_nd = [r["methods"]["CHymERA-Fast"]["KL_gapped"] for r in nondegenerate]
        rho = spearman_corr(eta_nd, kl_nd)
        p_rho = spearman_permutation_p_one_sided(eta_nd, kl_nd, n_perm=2000, seed=SEED + 200 + scale_idx)
        rho_frob = spearman_corr([r["eta_frob_max"] for r in nondegenerate], kl_nd)
        monotone = bool(rs) and all(r["closure_monotone"] for r in rs)
        endpoint_kl = max((r["methods"][tau_name(0.0)]["KL_gapped"] for r in rs), default=math.inf)
        clause_i = bool(len(dev_mech) == 18 and len(mech_c) == 36 and valid_count > 0 and assumption_invalid == 0 and violations == 0)
        clause_ii = bool(monotone)
        clause_iii = bool(endpoint_kl < 1e-6)
        clause_iv = bool(len(nondegenerate) >= 3 and rho > 0.0 and p_rho is not None and p_rho < 0.05)
        clause_v = bool(
            full_sample
            and (0.0 < tau_star < math.inf)
            and calibrated.get("mean_KL_gapped", math.inf) <= 0.05
            and (0.0 < calibrated.get("mean_replayed_layer_tokens", 0.0) < calibrated.get("sparse_mean_replayed_layer_tokens", 0.0))
        )
        if is_primary_gate_stride and is_primary_instruct:
            g4_pass = bool(full_sample and oom_c_count == 0 and clause_i and clause_ii and clause_iii and clause_iv and clause_v)
        else:
            g4_pass = None

        cells = {}
        for key in sorted({(r["task"],r["R"],r["T"]) for r in rs}):
            cell = [r for r in rs if (r["task"],r["R"],r["T"]) == key]
            cells[str(key)] = {"n": len(cell), "methods": aggregate_methods(cell),
                  **{field: float(np.mean([r[field] for r in cell])) for field in
                     ("prefill_base_ms","prefill_chymera_ms","prefill_wy_fold_ms","kv_evict_ms","extra_archive_mib","extra_archive_over_kv_ratio")}}
        drift = [{"instance_id": m["instance_id"], "split": m.get("split"), "T": m["T"], "task": m["task"],
                  "F": m["rel_err_chymera_F_by_layer"], "Fast": m["rel_err_chymera_Fast_by_layer"]} for m in mech_c]
        negative = [{"instance_id": r["instance_id"], "KL_gapped_F": r["methods"]["CHymERA-F"]["KL_gapped"],
                     "KL_gapped_Fast": r["methods"]["CHymERA-Fast"]["KL_gapped"]} for r in rs
                    if r["methods"]["CHymERA-F"]["KL_gapped"] > r["methods"]["CHymERA-Fast"]["KL_gapped"]]
        break_even = [{"instance_id": m["instance_id"], "split": m.get("split"), **m["break_even"]} for m in mc_all_c]
        dev_agg, test_agg = aggregate_methods(dev), aggregate_methods(test)
        g1a_local = bool(full_sample and all(comparisons[f"CHymERA-Fast vs {b}"]["local_pass"] for b in STALE_BASELINES))
        g1b_local = bool(full_sample and all(comparisons[f"CHymERA-F vs {b}"]["local_pass"] for b in STALE_BASELINES))
        summary["by_C"][str(C)] = {
            "gate_scope": gate_scope,
            "oom_count": oom_c_count,
            "n_dev": len(dev), "n_heldout_test": len(test), "full_preregistered_sample": full_sample,
            "methods_all": aggregate_methods(rs), "methods_heldout_test": test_agg, "G1_comparisons": comparisons,
            "tau_calibration": calibrated, "tau_curve_dev": {str(t): dev_agg.get(tau_name(t)) for t in TAU_SWEEP},
            "tau_curve_heldout_test": {str(t): test_agg.get(tau_name(t)) for t in TAU_SWEEP},
            "latency_memory_cells_task_R_T_C": cells,
            "gates": {
                "G1a": {"gate_scope": gate_scope, "method": "CHymERA-Fast", "pass_local": g1a_local if is_primary_gate_stride else None,
                        "pass": False if (is_primary_gate_stride and is_primary_instruct) else None,
                        "status": ("FAILED_INCOMPLETE_OR_OOM" if not full_sample else "FAILED_MISSING_MODELS_UNTIL_4SCALE_HOLM") if (is_primary_gate_stride and is_primary_instruct) else gate_scope},
                "G1b": {"gate_scope": gate_scope, "method": "CHymERA-F", "pass_local": g1b_local if is_primary_gate_stride else None,
                        "pass": False if (is_primary_gate_stride and is_primary_instruct) else None,
                        "status": ("FAILED_INCOMPLETE_OR_OOM" if not full_sample else "FAILED_MISSING_MODELS_UNTIL_4SCALE_HOLM") if (is_primary_gate_stride and is_primary_instruct) else gate_scope},
                "G2": {"gate_scope": gate_scope, "pass": g2_pass,
                       "subset_diagnostic_pass": g2_numeric if ms_taskb_heldout else None,
                       "complete_heldout_taskb_sample": complete_taskb,
                       "n_heldout_taskb_total": len(ms_taskb_heldout),
                       "n_heldout_short_island": len(ms_short_heldout),
                       "n_heldout_len1": n_heldout_l1, "n_heldout_len2": n_heldout_l2, "n_heldout_spill": n_heldout_spill,
                       "f_oracle_isolated_from_4A": isolated_ok,
                       "max_layer_rel_err_F": ef, "max_layer_rel_err_nohalo": en, "max_layer_rel_err_naive_halo": eh,
                       "dev_diagnostic": {"n_dev_short_island": len(ms_dev), "max_layer_rel_err_F": ef_dev,
                                          "max_layer_rel_err_nohalo": en_dev, "max_layer_rel_err_naive_halo": eh_dev}},
                "G3": {"gate_scope": gate_scope, "pass": g3_pass,
                       "zero_clean_suffix_calls_100pct": zero_calls,
                       "multiedit_numeric_completeness": multiedit_numeric_ok,
                       "symmetric_K4_Branching_T4096_heldout": k4_sym_cell,
                       "single_edit_T4096_R1_diagnostic": single_edit_diag},
                "G4": {"gate_scope": gate_scope, "pass": g4_pass,
                       "clauses": {"i_r23_and_prop8b_zero_violations": clause_i, "ii_monotone_closure": clause_ii,
                                   "iii_tau0_exact_endpoint": clause_iii, "iv_spearman_perm_positive_significant": clause_iv,
                                   "v_nontrivial_selective_gain": clause_v},
                       "n_mechanistic_checked": len(mech_c),
                       "r23_valid": valid_count, "assumption_invalid": assumption_invalid,
                       "prop8b_violations": violations, "monotone_100pct": monotone, "max_tau0_KL_gapped": endpoint_kl,
                       "spearman_eta_max_vs_KL_nondegenerate": rho, "spearman_perm_p_one_sided_2000": p_rho,
                       "spearman_perm_seed": SEED + 200 + scale_idx,
                       "spearman_frob_vs_KL_nondegenerate": rho_frob,
                       "n_nondegenerate": len(nondegenerate), "heldout_calibration_pass": clause_v},
                "G5": {"pass": True, "meaning": "transparent reporting obligation, never overrides G1-G4 failures",
                       "negative_result_F_worse_than_Fast": negative, "break_even": break_even, "layerwise_N_minus_F_drift": drift}}}
    return summary


def _run_joint_holm_m16(models, C_str="256"):
    """Joint M=16 step-down Holm-Bonferroni on Monte Carlo randomization / permutation p-values (B=2000, plus-one correction) across G1a (k=0..7) union G1b (k=8..15)."""
    M = 16
    missing_models = [tag for tag in PRIMARY_INSTRUCT_SCALES if tag not in models]
    any_oom_or_incomplete = False
    entries = []
    for m_group_idx, method_name in enumerate(("CHymERA-Fast", "CHymERA-F")):
        for s_idx, tag in enumerate(PRIMARY_INSTRUCT_SCALES):
            if tag not in models:
                any_oom_or_incomplete = True
                continue
            cell = models[tag].get("by_C", {}).get(C_str, {})
            if not cell.get("full_preregistered_sample", False) or cell.get("oom_count", 0) > 0:
                any_oom_or_incomplete = True
                continue
            comps = cell.get("G1_comparisons", {})
            for b_idx, baseline in enumerate(STALE_BASELINES):
                pair = f"{method_name} vs {baseline}"
                k_idx = 8 * m_group_idx + 2 * s_idx + b_idx
                st = comps.get(pair)
                if st is None or st.get("sign_flip_perm_p_one_sided") is None:
                    any_oom_or_incomplete = True
                    continue
                entries.append((k_idx, method_name, tag, baseline, pair, st["sign_flip_perm_p_one_sided"], st))
    ordered = sorted(entries, key=lambda x: (x[5], x[0]))
    per_comparison = {}
    running_adj_p = 0.0
    for rank_zero, (k_idx, method_name, tag, baseline, pair, p_raw, st) in enumerate(ordered):
        denom = M - rank_zero
        thresh = 0.05 / denom
        running_adj_p = max(running_adj_p, min(1.0, denom * p_raw))
        comp_pass = bool(
            not missing_models and not any_oom_or_incomplete and len(entries) == M
            and p_raw <= thresh and running_adj_p <= 0.05 and st["local_pass"]
        )
        per_comparison[f"{tag}::{pair}"] = {
            "k_index_M16": k_idx, "method": method_name, "model": tag, "baseline": baseline,
            "comparison": pair, "holm16_rank": rank_zero + 1, "holm16_alpha_threshold": thresh,
            "sign_flip_perm_raw_p": p_raw, "holm16_adjusted_p": running_adj_p,
            "median_ratio_pass": st["local_pass"], "pass": comp_pass,
        }
    if missing_models:
        status, g1a_pass, g1b_pass = "FAILED_MISSING_MODELS", False, False
    elif any_oom_or_incomplete or len(entries) < M:
        status, g1a_pass, g1b_pass = "EVALUATED_INCOMPLETE_OR_OOM_FAIL", False, False
    else:
        status = "EVALUATED"
        g1a_pass = bool(all(v["pass"] for v in per_comparison.values() if v["k_index_M16"] < 8))
        g1b_pass = bool(all(v["pass"] for v in per_comparison.values() if v["k_index_M16"] >= 8))
    return {
        "M": M, "C": int(C_str), "status": status, "missing_models": missing_models,
        "G1a_pass": bool(g1a_pass), "G1b_pass": bool(g1b_pass), "comparisons": per_comparison,
    }


def _verify_session_hardware_lock(out_dir: Path, models: dict) -> dict | None:
    """Enforce immutable session_hardware_lock.json across all non-tiny models and final aggregation."""
    lock_path = out_dir / "session_hardware_lock.json"
    non_tiny_models = {tag: m for tag, m in models.items() if not m.get("tiny_arch_dryrun", False)}
    if not non_tiny_models and not lock_path.exists():
        return None
    if not lock_path.is_file():
        raise RuntimeError(f"MISSING_SESSION_HARDWARE_LOCK: {lock_path} is required for real-weight aggregation")
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    locked_uuid = str(lock_data.get("cuda_uuid", "")).strip()
    if not locked_uuid:
        raise RuntimeError(f"EMPTY_LOCKED_CUDA_UUID in {lock_path}")
    for tag, m in non_tiny_models.items():
        m_uuid = str((m.get("gpu_properties") or {}).get("cuda_uuid", "")).strip()
        if not m_uuid:
            raise RuntimeError(f"EMPTY_MODEL_CUDA_UUID in summary_model.json for {tag}")
        if m_uuid != locked_uuid:
            raise RuntimeError(f"SINGLE_SESSION_GPU_VIOLATION for {tag}: {m_uuid} != locked {locked_uuid}")
    return lock_data


def aggregate_completed_models(out_dir):
    """No torch/model imports: supports independently completed model runs with strict hardware lock & False-on-missing."""
    out_dir = Path(out_dir)
    models = {}
    model_paths = {}
    for path in sorted(out_dir.glob("model_*/summary_model.json")):
        model = json.loads(path.read_text())
        if model.get("status") in ("COMPLETE", "COMPLETE_WITH_OOM", "TRUNCATED_INCOMPLETE", "FAILED_EXCEPTION"):
            models[model["model"]] = model
            model_paths[model["model"]] = path
    lock_data = _verify_session_hardware_lock(out_dir, models)
    joint_holm = _run_joint_holm_m16(models, "256")
    for tag in PRIMARY_INSTRUCT_SCALES:
        cell = models.get(tag, {}).get("by_C", {}).get("256")
        if cell:
            for gate_key, fam_pass, method_name in (
                ("G1a", joint_holm["G1a_pass"], "CHymERA-Fast"),
                ("G1b", joint_holm["G1b_pass"], "CHymERA-F"),
            ):
                model_keys = [f"{tag}::{method_name} vs {b}" for b in STALE_BASELINES]
                model_pass = bool(
                    fam_pass
                    and all(k in joint_holm["comparisons"] and joint_holm["comparisons"][k]["pass"] for k in model_keys)
                )
                cell["gates"][gate_key].update(
                    status=("FAILED_INCOMPLETE_OR_OOM"
                            if (not cell.get("full_preregistered_sample", False) or cell.get("oom_count", 0) > 0)
                            else joint_holm["status"]),
                    family_pass_joint_M16=bool(fam_pass),
                    model_pair_pass=model_pass,
                    **{"pass": bool(fam_pass)},
                )
            if tag in model_paths:
                write_json(model_paths[tag], models[tag])
    all_four_present = all(
        tag in models
        and models[tag].get("status") == "COMPLETE"
        and not models[tag].get("truncated", False)
        and "256" in models[tag].get("by_C", {})
        for tag in PRIMARY_INSTRUCT_SCALES
    )
    overall_c256 = {
        "G1a": bool(joint_holm["G1a_pass"]),
        "G1b": bool(joint_holm["G1b_pass"]),
        "G2": bool(all_four_present and all(bool(models[t]["by_C"]["256"]["gates"]["G2"]["pass"]) for t in PRIMARY_INSTRUCT_SCALES)),
        "G3": bool(all_four_present and all(bool(models[t]["by_C"]["256"]["gates"]["G3"]["pass"]) for t in PRIMARY_INSTRUCT_SCALES)),
        "G4": bool(all_four_present and all(bool(models[t]["by_C"]["256"]["gates"]["G4"]["pass"]) for t in PRIMARY_INSTRUCT_SCALES)),
        "G5": True,
    }
    disclosures = [{"model":tag,"C":C, **cell["gates"]["G5"]} for tag,m in models.items() for C,cell in m["by_C"].items()]
    report = {"version":"R2.1", "primary_gate_stride": 256,
              "session_hardware_lock": lock_data,
              "overall_gates_C256": overall_c256,
              "G1_Joint_Holm_M16_SignFlipPerm_C256": joint_holm,
              "mandatory_negative_result_disclosures": disclosures,
              "models": models,
              "scaling_law": {tag:{C:{"per_cell":cell["latency_memory_cells_task_R_T_C"],
                                     "layerwise_drift":cell["gates"]["G5"]["layerwise_N_minus_F_drift"]}
                                  for C,cell in m["by_C"].items()} for tag,m in models.items() if tag in PRIMARY_INSTRUCT_SCALES}}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir/"summary_r2.json",report)
    write_json(out_dir/"summary_all.json",report)
    return report

def make_tiny_model():
    """Actual 8-layer, two-block Qwen3_5ForConditionalGeneration, random weights."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    text = dict(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=8,
                num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                linear_num_key_heads=4, linear_num_value_heads=8,
                linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
                layer_types=["linear_attention"]*3+["full_attention"] + ["linear_attention"]*3+["full_attention"],
                max_position_embeddings=8192, pad_token_id=0, eos_token_id=1,
                rope_parameters={"rope_type":"default", "rope_theta":10000.0,
                                 "partial_rotary_factor":0.25, "mrope_section":[1,1,2]})
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=128,
                  patch_size=2, temporal_patch_size=1, spatial_merge_size=1, num_position_embeddings=16)
    cfg = Qwen3_5Config(text_config=text, vision_config=vision)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    return Qwen3_5ForConditionalGeneration(cfg).eval()


class TinyTokenizer:
    """Synthetic IDs only for random-weight architecture tests, never BPE tests."""
    eos_token_id = 1
    def encode(self, text, add_special_tokens=False):
        return [2 + ord(c) for c in text]
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr((i-2)%95+32) for i in ids)


def tiny_instances():
    rows = []
    for i, spans in enumerate(([(17,25),(27,35)], [(68,99)])):
        geom = build_survivor_geometry(spans,128)
        rows.append(dict(instance_id=i, task="TaskB_RAGEvict" if i==0 else "TaskC_SlidingScrub",
             sub_regime="short_island_len1" if i==0 else "pinned_sliding_window", split="dev", T=128,
             body_ids=[5+(j*17+i*11)%480 for j in range(128)], query_ids=[7,11,13],
             spans=spans, v_gold="amber-1234-anchor", v_stale="cobalt-5678-ledger",
             P_pin=68 if i else None, W_keep=100 if i else None,
             L_prime=geom["L_prime"], halo_size=len(geom["halo_indices"]), island_lengths=geom["island_lengths"]))
    return rows


def run_gpu_benchmark(out_dir, model_tags, chunk_stride=None, max_instances=None, wall_budget_s=DEFAULT_WALL_BUDGET_S,
                      chunk_strides=None, device="auto", tiny_arch_dryrun=False):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strides = list(dict.fromkeys(chunk_strides or ([chunk_stride] if chunk_stride is not None else [256,1024])))
    if any(c < CONV_W or c%64 for c in strides):
        raise ValueError("--chunk-strides requires positive multiples of 64")
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    gpu_props = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        cuda_uuid = str(getattr(props, "uuid", "")).strip()
        if not cuda_uuid:
            raise RuntimeError("EMPTY_CUDA_UUID: torch.cuda.get_device_properties(device).uuid is empty on CUDA")
        gpu_props = {"name": props.name, "total_memory_mib": props.total_memory / (2**20), "cuda_uuid": cuda_uuid}
        if not tiny_arch_dryrun:
            if "A100" not in props.name or props.total_memory < 38 * (2**30):
                raise RuntimeError(f"HARDWARE_TIER_MISMATCH: registered real-weight runs require A100 (>=38 GiB), got {gpu_props}")
            lock_path = out_dir / "session_hardware_lock.json"
            if lock_path.exists():
                lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
                locked_uuid = str(lock_data.get("cuda_uuid", "")).strip()
                if not locked_uuid or locked_uuid != cuda_uuid:
                    raise RuntimeError(
                        f"SINGLE_SESSION_GPU_VIOLATION: {cuda_uuid} != locked {locked_uuid} in {lock_path}"
                    )
            else:
                write_json(lock_path, {
                    "locked_at_unix": time.time(),
                    "cuda_uuid": cuda_uuid,
                    "name": props.name,
                    "total_memory_mib": gpu_props["total_memory_mib"],
                })
            prior_summaries = list(out_dir.glob("model_*/summary_model.json"))
            for p_sum in prior_summaries:
                prev_uuid = str((json.loads(p_sum.read_text()).get("gpu_properties") or {}).get("cuda_uuid", "")).strip()
                if not prev_uuid or prev_uuid != cuda_uuid:
                    raise RuntimeError(f"SINGLE_SESSION_GPU_VIOLATION: {cuda_uuid} != {prev_uuid} from {p_sum}")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    source_sha = hashlib.sha256(Path(qm.__file__).read_bytes()).hexdigest()
    if not tiny_arch_dryrun and source_sha != LOCKED_MODELING_SHA256:
        raise RuntimeError("LOCKED_MODELING_SHA_MISMATCH")
    start = time.perf_counter()
    for tag in model_tags:
        if tag not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model tag {tag}")
        spec = MODEL_REGISTRY[tag]
        mdir = out_dir/f"model_{tag}"
        mdir.mkdir(exist_ok=True)
        if tiny_arch_dryrun:
            tokenizer, prompts, full = TinyTokenizer(), tiny_instances(), make_tiny_model()
        else:
            tokenizer = AutoTokenizer.from_pretrained(spec["repo_id"], revision=spec["revision"])
            # Validate all 96 prompts before a limit is applied or weights are loaded.
            instruct = build_benchmark_dataset(tokenizer, True)
            base = build_benchmark_dataset(tokenizer, False)
            assert len(instruct)==72 and len(base)==24
            prompts = instruct if spec["is_instruct"] else base
            full = Qwen3_5ForConditionalGeneration.from_pretrained(spec["repo_id"], revision=spec["revision"],
                                                  dtype=dtype, attn_implementation="eager")
        if max_instances is not None:
            prompts = prompts[:max_instances]
        full = full.eval().to(device=device, dtype=dtype)
        cfg = full.config.text_config
        if not tiny_arch_dryrun:
            exp = spec["expected"]
            actual = dict(layers=cfg.num_hidden_layers, h_k=cfg.linear_num_key_heads,
                          h_v=cfg.linear_num_value_heads, h_kv=cfg.num_key_value_heads, d_model=cfg.hidden_size)
            assert actual == exp and cfg.head_dim == 256, "ARCH_MISMATCH"
            assert cfg.vocab_size == 248320, "VOCAB_SIZE_MISMATCH"
            assert cfg.linear_key_head_dim == 128 and cfg.linear_value_head_dim == 128 and cfg.linear_conv_kernel_dim == CONV_W
            assert list(cfg.layer_types) == (["linear_attention"]*3+["full_attention"])*(cfg.num_hidden_layers//4), "LAYER_TYPES_MISMATCH"
        engine = HybridBenchEngine(full, tokenizer, device)
        rows, mech, multi, oom_events = [], [], [], []
        complete = True
        try:
            with torch.inference_mode():
                # Conv lag ordering and unrecorded warmup.
                w = torch.tensor([[10.,20.,30.,40.]],device=device)
                x = torch.tensor([[[1.,0.,0.,0.]]],device=device)
                y = qm.causal_conv1d_fn(x,w,None,activation=None)
                assert torch.allclose(y.flatten(),torch.tensor([40.,30.,20.,10.],device=device)), "CONV_LAG_MISMATCH"
                if prompts:
                    engine.forward(engine.tensor(prompts[0]["body_ids"][:64]), positions=engine.positions(64))
                for idx, prompt in enumerate(prompts):
                    if time.perf_counter()-start > wall_budget_s:
                        complete = False
                        break
                    shared = None
                    prompt = {**prompt, "model":tag}
                    for C in strides:
                        try:
                            result, mechanism, shared, archive = evaluate_instance(engine,prompt,C,shared,tiny_arch_dryrun)
                            rows.append(result)
                            if mechanism:
                                mech.append(mechanism)
                            if prompt["task"] == "TaskC_SlidingScrub" and (prompt["T"] == 4096 or tiny_arch_dryrun):
                                multi.append(benchmark_task_c_multiedit(
                                    engine, prompt, C, archive, result["prefill_chymera_ms"], result["prefill_wy_fold_ms"]
                                ))
                            del archive
                        except torch.cuda.OutOfMemoryError as oom_exc:
                            engine.mode, engine.phase, engine.f_oracle_buffer = "off", None, None
                            engine.h, engine.ops, engine.ph, engine.q, engine.gamma = {}, {}, {}, {}, {}
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            oom_rec = {"model": tag, "instance_id": prompt["instance_id"], "task": prompt["task"],
                                       "split": prompt["split"], "T": prompt["T"], "C": C,
                                       "status": "OOM", "error": str(oom_exc)}
                            oom_events.append(oom_rec)
                            print(f"[OOM] {oom_rec}", flush=True)
                    del shared
                    if (idx+1)%8 == 0 or idx+1 == len(prompts):
                        write_jsonl(mdir/"instances_partial.jsonl",rows)
                        write_jsonl(mdir/"mechanistic_partial.jsonl",mech)
                        write_json(mdir/"task_c_multiedit_partial.json",multi)
                        if oom_events:
                            write_json(mdir/"oom_events_partial.json",oom_events)
                        print(f"[{tag}] {idx+1}/{len(prompts)} instances, C={strides}, ooms={len(oom_events)}",flush=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            write_jsonl(mdir/"instances_partial.jsonl",rows)
            expected_n = len(tiny_instances()) if tiny_arch_dryrun else (72 if spec["is_instruct"] else 24)
            is_truncated = (not complete) or (len(prompts) < expected_n)
            if is_truncated:
                write_json(mdir/"run_incomplete.json", {
                    "status": "CENSORED_WALL_BUDGET_OR_MAX_INSTANCES",
                    "n_instances": len({r["instance_id"] for r in rows}),
                    "expected_instances": expected_n,
                    "oom_events": oom_events,
                })
            write_jsonl(mdir/"instances.jsonl",rows)
            write_jsonl(mdir/"mechanistic_dev.jsonl",mech)
            write_json(mdir/"task_c_multiedit.json",multi)
            if oom_events:
                write_json(mdir/"oom_events.json",oom_events)
            report = summarize_model(
                tag, rows, mech, multi, tiny_arch_dryrun, oom_events=oom_events, truncated=is_truncated
            )
            for C_key, cell in report["by_C"].items():
                g5 = cell["gates"]["G5"]
                if g5["negative_result_F_worse_than_Fast"]:
                    print(f"[{tag}, C={C_key}] G5: F has larger N_gapped KL than Fast on {len(g5['negative_result_F_worse_than_Fast'])} rows; exact KL pairs and layerwise N-F drift are in summary_model.json",flush=True)
                for entry in g5["break_even"]:
                    for method in ("CHymERA-F", "CHymERA-Fast"):
                        item = entry[method]["Sparse-Anchor-SuffixReplay"]
                        if item["K_star"] > 4:
                            print(f"[{tag}, C={C_key}] G5: instance={entry['instance_id']} {method} branching K_star={item['K_star']} (>4) vs Sparse-Anchor-SuffixReplay",flush=True)
            exp = spec["expected"]
            report.update(modeling_sha256=source_sha, locked_modeling_sha256=LOCKED_MODELING_SHA256,
                 cpu_snapshot_operator_sha256=getattr(qm,"_r21_snapshot_sha",None),
                 model_revision=spec["revision"], device=str(device), gpu_properties=gpu_props,
                 memory_accounting_T4096={str(C):compute_exact_memory_bytes(exp,4096,C) for C in strides},
                 configuration_sha256=hashlib.sha256(cfg.to_json_string().encode()).hexdigest())
            write_json(mdir/"summary_model.json",report)
            aggregate_completed_models(out_dir)
        except Exception as exc:
            fail_report = summarize_model(
                tag, rows, mech, multi, tiny_arch_dryrun, oom_events=oom_events, truncated=True
            )
            fail_report["status"] = "FAILED_EXCEPTION"
            fail_report["fatal_error"] = str(exc)
            fail_report.update(modeling_sha256=source_sha, locked_modeling_sha256=LOCKED_MODELING_SHA256,
                 model_revision=spec["revision"], device=str(device), gpu_properties=gpu_props)
            write_json(mdir/"summary_model.json", fail_report)
            try:
                aggregate_completed_models(out_dir)
            except Exception:
                pass
            raise
        finally:
            engine.close()
            del engine,full
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return aggregate_completed_models(out_dir)


@contextmanager
def locked_cpu_operator_context(qm):
    """Run snapshot operators in a compatible installed Qwen CPU architecture.

    The installed package supplies config/model infrastructure. The exact local
    locked forward/conv/recurrence bodies supply the operators under test. No
    package or repository file is edited; all replacements are restored.
    """
    source = Path(__file__).resolve().parent/"research_sources/modeling_qwen3_5.py"
    if not source.exists():
        yield
        return
    data = source.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    assert sha == LOCKED_MODELING_SHA256, "LOCAL_MODELING_SHA_MISMATCH"
    tree = ast.parse(data)
    saved = []
    namespace = qm.__dict__
    operators = {"l2norm", "causal_conv1d_fn", "causal_conv1d_update",
                 "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"}
    classes = {"Qwen3_5GatedDeltaNet", "Qwen3_5TextModel", "Qwen3_5DecoderLayer"}
    try:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in operators:
                node.decorator_list = []
                saved.append((qm, node.name, getattr(qm,node.name)))
                exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),"exec"),namespace)
            elif isinstance(node, ast.ClassDef) and node.name in classes:
                forward = copy.deepcopy(next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=="forward"))
                forward.decorator_list = []
                local = {}
                exec(compile(ast.fix_missing_locations(ast.Module(body=[forward],type_ignores=[])),str(source),"exec"),namespace,local)
                cls = getattr(qm,node.name)
                saved.append((cls,"forward",cls.forward))
                cls.forward = local["forward"]
        qm._r21_snapshot_sha = sha
        print("[self-test] Executing locked local Qwen forward/conv/chunk/recurrent operator bodies",flush=True)
        yield
    finally:
        for obj,name,value in reversed(saved):
            setattr(obj,name,value)
        qm.__dict__.pop("_r21_snapshot_sha",None)


def run_tiny_arch_self_test():
    try:
        import torch
    except ImportError:
        print("[self-test] SKIP tiny PyTorch forward: torch is not installed (algebra/BPE checks still required)",flush=True)
        return
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError as exc:
        raise RuntimeError("torch is installed; install Qwen3.5-capable transformers to execute the required end-to-end dry-run") from exc
    torch.set_num_threads(1)
    # Independent oracle for the exact GPU helper, including 2:1 key sharing
    # and a non-native-size tail, before exercising the entire model pipeline.
    gen = torch.Generator().manual_seed(SEED)
    k = torch.randn(1, 140, 4, 32, generator=gen)
    v = torch.randn(1, 140, 8, 32, generator=gen)
    g = -torch.rand(1, 140, 8, generator=gen)*0.05
    beta = torch.rand(1, 140, 8, generator=gen)*0.9
    P, H = torch_wy_chunk_summary(k,v,g,beta)
    kr = k.repeat_interleave(2,dim=2)
    kr = kr * torch.rsqrt(kr.square().sum(-1,keepdim=True)+EPS)
    pn, hn = np_wy_chunk_summary(kr[0].transpose(0,1).double().numpy(),
                               v[0].transpose(0,1).double().numpy(),g[0].T.double().numpy(),beta[0].T.double().numpy())
    assert np.max(np.abs(P.numpy()-pn)) < 1e-5
    assert np.max(np.abs(H.numpy()-hn)) < 1e-5
    with locked_cpu_operator_context(modeling_qwen3_5):
        with tempfile.TemporaryDirectory(prefix="chymera-r21-") as tmp:
            report = run_gpu_benchmark(Path(tmp),["0.8B"],chunk_strides=[64,128],device="cpu",
                                       tiny_arch_dryrun=True,wall_budget_s=1200)
            folder = Path(tmp)/"model_0.8B"
            for name in ("instances.jsonl","mechanistic_dev.jsonl","task_c_multiedit.json","summary_model.json"):
                assert (folder/name).exists(), name
            rows = [json.loads(line) for line in (folder/"instances.jsonl").read_text().splitlines()]
            mech = [json.loads(line) for line in (folder/"mechanistic_dev.jsonl").read_text().splitlines()]
            multi = json.loads((folder/"task_c_multiedit.json").read_text())
            assert len(rows)==4 and len(mech)==4 and len(multi)==2
            for row in rows:
                assert len(row["methods"]) == 6+len(TAU_SWEEP)
                assert row["full_closure_cache_max_abs_diff"] < 1e-6
                assert math.isfinite(row["direct_vs_delta_max_abs_err"]) and row["direct_vs_delta_rel_err"] <= 1e-4
                assert row["closure_monotone"]
                for name in ("CHymERA-F","CHymERA-Fast"):
                    assert row["methods"][name]["clean_suffix_fa_mlp_calls"] == 0
                assert row["methods"]["CHymERA-F"]["pass1_audit"]["f_oracle_buffer_was_none"] is True
                assert row["methods"][tau_name(0.0)]["KL_gapped"] == 0
                assert row["methods"][tau_name(math.inf)]["KL_gapped"] == row["methods"]["CHymERA-Fast"]["KL_gapped"]
                assert row["methods"][tau_name(math.inf)]["edit_latency_ms"] == row["methods"]["CHymERA-Fast"]["edit_latency_ms"]
            for m in mech:
                assert m["f_oracle_isolated_from_4A"] is True
                assert max(m["rel_err_chymera_F_by_layer"].values()) < 1e-3
                assert m["rel_err_chymera_Fast_by_layer"]["0"] < 1e-3
                assert "Terminal-Inverse" in m["ablations"]
                assert len(m["blockwise_closure_trajectory"]) == 3
                assert all(x["prop8b_violations"]==0 for x in m["real_weight_adjoint"].values())
            for protocol in (p for entry in multi for p in entry["protocols"]):
                expected = protocol["K"]-1 if protocol["protocol"]=="Cumulative-Rebuild" and protocol["method"]!="Full-Prefill-Compact" else 0
                assert protocol["archive_rebuilds"] == expected
            # Stride-invariant baseline results and timings must be reused exactly.
            for i in (0,1):
                a,b = [r for r in rows if r["instance_id"]==i]
                for name in ("KV-Only-Evict","Full-Prefill-Compact"):
                    for metric in ("pred","edit_latency_ms","decode_ms","EM_gold"):
                        assert a["methods"][name][metric] == b["methods"][name][metric]
            assert any(r["methods"]["Sparse-Anchor-SuffixReplay"]["clean_suffix_fa_mlp_calls"] > 0 for r in rows)
            again = aggregate_completed_models(Path(tmp))
            json.dumps(sanitize_json({"torch":torch.tensor([1.0,float("inf")]),"report":again}),allow_nan=False)
            assert report["models"].keys() == again["models"].keys()
            # Full four-model preregistered gate verification (G1a, G1b, G2, G3, G4 + vacuous/OOM negative tests).
            stats_dir = Path(tmp)/"statistics_fixture"
            fixture_rows, fixture_mech, fixture_multi = [], [], []
            for i in range(72):
                split = "dev" if i < 18 else "heldout_test"
                for c_val in (256, 1024):
                    r = copy.deepcopy(rows[0])
                    r.update(instance_id=i, C=c_val, T=4096, R=1, split=split, eta_adj_max=0.1+i*0.002)
                    for name, m in r["methods"].items():
                        m["EM_gold"] = 1
                        m["clean_suffix_fa_mlp_calls"] = 0
                        m["KL_gapped"] = 0.05 + i * 0.001
                        m["edit_latency_ms"] = 1.0 if name == "CHymERA-Fast" else 3.0
                        m["replayed_layer_tokens"] = 500 if name == "CHymERA-Fast" else 4000
                        if name in ("KV-Only-Evict", "CLM-NearestAnchorStaleGDN"):
                            m["KL_gapped"] = 3.0 + i * 0.02
                    # Make tau=inf fail dev <= 0.02 (KL=0.06), tau=0.05 pass dev <= 0.02 (KL=0.01) with strict 0 < tokens < sparse!
                    for tau in TAU_SWEEP:
                        m_t = r["methods"][tau_name(tau)]
                        m_t["unrefreshed_clean_suffix_fa_mlp_calls"] = 0
                        if math.isinf(tau):
                            m_t["KL_gapped"] = 0.06
                            m_t["replayed_layer_tokens"] = 500
                        elif tau == 0.0:
                            m_t["KL_gapped"] = 0.0
                            m_t["replayed_layer_tokens"] = 4000
                        else:
                            m_t["KL_gapped"] = 0.01
                            m_t["replayed_layer_tokens"] = 1500
                    fixture_rows.append(r)
                if split == "dev":
                    mc_dev = copy.deepcopy(mech[0])
                    mc_dev.update(instance_id=i, C=256, split="dev", sub_regime="single_span")
                    mc_dev["real_weight_adjoint"] = {"0": {"r23_valid": 8, "assumption_invalid": 0, "prop8b_violations": 0}}
                    mc_dev["pure_suffix_adjoint"] = {"0": {"r23_valid": 8, "assumption_invalid": 0, "prop8b_violations": 0}}
                    fixture_mech.append(mc_dev)
            for j in range(18):
                mc_h = copy.deepcopy(mech[0])
                sub_reg = ("short_island_len1" if j < 6 else "short_island_len2" if j < 12 else "chunk_boundary_halo_spill")
                mc_h.update(instance_id=18+j, C=256, split="heldout_test", task="TaskB_RAGEvict",
                            sub_regime=sub_reg,
                            f_oracle_isolated_from_4A=True,
                            rel_err_chymera_F_by_layer={"0": 1e-5},
                            rel_err_nohalo_by_layer={"0": 0.05},
                            rel_err_naive_halo_by_layer={"0": 0.05} if j < 12 else {})
                mc_h["real_weight_adjoint"] = {}
                mc_h["pure_suffix_adjoint"] = {"0": {"r23_valid": 8, "assumption_invalid": 0, "prop8b_violations": 0}}
                fixture_mech.append(mc_h)
            for k_i in range(6):
                for c_val in (256, 1024):
                    me = copy.deepcopy(multi[0])
                    me.update(instance_id=60+k_i, split="heldout_test", T=4096, C=c_val)
                    for p in me["protocols"]:
                        if p["protocol"] == "Branching-Counterfactual" and p["K"] == 4:
                            p["mean_branch_EM_gold"] = 1.0
                            p["symmetric_total_ms"] = (10.0 if p["method"] == "CHymERA-Fast"
                                                       else 15.0 if p["method"] == "Sparse-Anchor-SuffixReplay"
                                                       else 20.0)
                    fixture_multi.append(me)
            stats_dir.mkdir(parents=True, exist_ok=True)
            mock_gpu_props = {"name": "NVIDIA A100-SXM4-40GB", "total_memory_mib": 40960.0, "cuda_uuid": "GPU-MOCK-A100-UUID-0001"}
            write_json(stats_dir/"session_hardware_lock.json", {
                "locked_at_unix": 0.0, **mock_gpu_props
            })
            for tag in PRIMARY_INSTRUCT_SCALES:
                md = stats_dir/f"model_{tag}"
                md.mkdir(parents=True)
                s_mod = summarize_model(tag, fixture_rows, fixture_mech, fixture_multi)
                s_mod["gpu_properties"] = dict(mock_gpu_props)
                write_json(md/"summary_model.json", s_mod)
            stats = aggregate_completed_models(stats_dir)
            assert stats["overall_gates_C256"] == {"G1a": True, "G1b": True, "G2": True, "G3": True, "G4": True, "G5": True}
            assert (stats_dir/"summary_all.json").exists()
            for model in stats["models"].values():
                assert model["by_C"]["1024"]["gate_scope"] == "REPORTING_ONLY_NON_SELECTABLE"
                assert model["by_C"]["1024"]["gates"]["G2"]["pass"] is None
            # Negative test 1: vacuous tau_dev_star == inf MUST fail G4(v)!
            vac_rows = copy.deepcopy(fixture_rows)
            for r in vac_rows:
                r["methods"][tau_name(math.inf)]["KL_gapped"] = 0.005
            vac_sum = summarize_model("0.8B", vac_rows, fixture_mech, fixture_multi)
            assert vac_sum["by_C"]["256"]["tau_calibration"]["tau_dev_star"] == math.inf
            assert vac_sum["by_C"]["256"]["gates"]["G4"]["pass"] is False
            # Negative test 2: OOM event MUST fail ALL gates G1a, G1b, G2, G3, G4!
            oom_sum = summarize_model("0.8B", fixture_rows, fixture_mech, fixture_multi, oom_events=[{"C": 256, "status": "OOM"}])
            for gk in ("G1a", "G1b", "G2", "G3", "G4"):
                assert oom_sum["by_C"]["256"]["gates"][gk]["pass"] is False, gk
            # Negative test 2b: Missing C=1024 multi-edit rows or C=1024 OOM MUST fail G3(ii) at C=256!
            multi_no_1024 = [m for m in fixture_multi if m["C"] != 1024]
            g3_no1024_sum = summarize_model("0.8B", fixture_rows, fixture_mech, multi_no_1024)
            assert g3_no1024_sum["by_C"]["256"]["gates"]["G3"]["pass"] is False
            g3_oom1024_sum = summarize_model("0.8B", fixture_rows, fixture_mech, fixture_multi, oom_events=[{"C": 1024, "status": "OOM"}])
            assert g3_oom1024_sum["by_C"]["256"]["gates"]["G3"]["pass"] is False
            # Negative test 3: Incomplete 4-scale family MUST evaluate to False (never None/PENDING)!
            (stats_dir/"model_9B"/"summary_model.json").unlink()
            incomplete = aggregate_completed_models(stats_dir)
            for gk in ("G1a", "G1b", "G2", "G3", "G4"):
                assert incomplete["overall_gates_C256"][gk] is False, gk
            assert all(m["by_C"]["256"]["gates"]["G1a"]["pass"] is False for m in incomplete["models"].values())
            assert all(m["by_C"]["256"]["gates"]["G1b"]["pass"] is False for m in incomplete["models"].values())
            # Negative test 4: Truncated run MUST evaluate G1a..G4 to False!
            trunc_sum = summarize_model("0.8B", fixture_rows, fixture_mech, fixture_multi, truncated=True)
            for gk in ("G1a", "G1b", "G2", "G3", "G4"):
                assert trunc_sum["by_C"]["256"]["gates"][gk]["pass"] is False, gk
            # Negative test 5: Missing or mismatched session_hardware_lock.json MUST raise RuntimeError!
            write_json(stats_dir/"session_hardware_lock.json", {"cuda_uuid": "GPU-WRONG-UUID"})
            try:
                aggregate_completed_models(stats_dir)
                raise AssertionError("Expected SINGLE_SESSION_GPU_VIOLATION")
            except RuntimeError as exc:
                assert "SINGLE_SESSION_GPU_VIOLATION" in str(exc)
            (stats_dir/"session_hardware_lock.json").unlink()
            try:
                aggregate_completed_models(stats_dir)
                raise AssertionError("Expected MISSING_SESSION_HARDWARE_LOCK")
            except RuntimeError as exc:
                assert "MISSING_SESSION_HARDWARE_LOCK" in str(exc)
            # Negative test 6: verify_prereg_manifest pass + tamper abort!
            prereg_dir = Path(tmp)/"prereg_test"
            prereg_dir.mkdir(parents=True, exist_ok=True)
            manifest_lines = []
            for fname in PREREG_REQUIRED_FILES:
                fp = prereg_dir/fname
                fp.write_text(f"fixture-{fname}\n", encoding="utf-8")
                manifest_lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {fname}")
            mfile = prereg_dir/"PREREG_R2.sha256"
            mfile.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
            verify_prereg_manifest(mfile)
            (prereg_dir/"colab_bench_r2.py").write_text("tampered\n", encoding="utf-8")
            try:
                verify_prereg_manifest(mfile)
                raise AssertionError("Expected PREREG_SHA256_MISMATCH")
            except RuntimeError as exc:
                assert "PREREG_SHA256_MISMATCH" in str(exc)
            # Explicitly execute the CLI aggregation path in an independent process.
            subprocess.check_call([sys.executable,__file__,"--aggregate-only","--out-dir",tmp])
    print("[self-test] Actual tiny Qwen3.5 CPU pipeline: all system/ablation methods, closure, multi-edit, checkpoint and aggregation PASS",flush=True)


def main():
    parser = argparse.ArgumentParser(description="R2.1 CHymERA real-weight benchmark")
    parser.add_argument("--verify-prereg",default=None,help="Path to PREREG_R2.sha256 manifest to verify at startup")
    parser.add_argument("--cpu-self-test",action="store_true")
    parser.add_argument("--tiny-arch-dryrun",action="store_true")
    parser.add_argument("--aggregate-only",action="store_true")
    parser.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    parser.add_argument("--out-dir",default="results_r2/bench_r2_out")
    parser.add_argument("--models",default=",".join(MODEL_REGISTRY))
    parser.add_argument("--chunk-strides",default=None,help="Comma-separated strides (default 256,1024)")
    parser.add_argument("--chunk-stride",type=int,default=None,help="Legacy single-stride alias")
    parser.add_argument("--max-instances",type=int,default=None)
    parser.add_argument("--wall-budget-s",type=float,default=DEFAULT_WALL_BUDGET_S)
    parser.add_argument("--deps-ready",action="store_true")
    args = parser.parse_args()
    if args.verify_prereg:
        verify_prereg_manifest(args.verify_prereg)
    if args.aggregate_only:
        report = aggregate_completed_models(Path(args.out_dir))
        print(f"[aggregate] {len(report['models'])} completed models -> summary_r2.json & summary_all.json",flush=True)
        return
    if args.cpu_self_test:
        run_cpu_self_test()
        return
    if args.chunk_strides and args.chunk_stride is not None:
        parser.error("choose --chunk-strides or --chunk-stride")
    strides = ([int(c.strip()) for c in args.chunk_strides.split(",")] if args.chunk_strides else
               [args.chunk_stride] if args.chunk_stride else [64] if args.tiny_arch_dryrun else [256,1024])
    if not args.tiny_arch_dryrun:
        ensure_transformers_installed(allow_install=not args.deps_ready)
    run_gpu_benchmark(Path(args.out_dir),[t.strip() for t in args.models.split(",")],chunk_strides=strides,
                      max_instances=args.max_instances,wall_budget_s=args.wall_budget_s,
                      device=args.device,tiny_arch_dryrun=args.tiny_arch_dryrun)


if __name__ == "__main__":
    main()
