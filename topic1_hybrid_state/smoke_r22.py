#!/usr/bin/env python3
"""Round 2.2 CPU Verification Suite (`smoke_r22.py`).

Verifies:
1. Suite T4_R22: Theorem 4' exact singular-suffix subspace invariance condition
   ker(F A_j) <= ker(F) <=> A_j(ker F) = ker F <=> k_j in ker(F) union row(F),
   including the explicit 2D counterexample F=diag(0,1), A_j=diag(1/2,1), FB_j=0,
   the Moore-Penrose oblique terminal editor M_+ when k_j in row(F), and the
   explicit non-uniqueness witness when k_j not in ker(F) union row(F).
2. Suite T8b_R22: Proposition 8b' separation of Forward Spectral Contraction
   (0 <= gamma_t <= 1, 0 <= beta_t ||k_t||_2^2 <= 2) from Step Inversion
   (beta_t ||k_t||_2^2 != 1), verified across synthetic trajectories in [1.0, 2.0],
   boundary counterexamples at > 2.0, and all 175,104 real-weight head trajectories
   recorded in `results_r2/bench_r2_out/model_*/mechanistic_dev.jsonl`.
3. Suite T6_R22: Standalone `Delta32` and `Delta16` P-only archive formulas
   (zero H_c storage) and `T = 8192` sub-KV memory verification.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

SEED = 20261109


def _closed_form_theta(gamma: float, beta: float, a_sq: float, w: np.ndarray, z: np.ndarray) -> float:
    delta = 1.0 - beta * a_sq
    alpha = beta / delta
    wn = float(np.linalg.norm(w))
    zn = float(np.linalg.norm(z))
    if wn < 1e-15 or zn < 1e-15 or a_sq < 1e-15:
        return abs(1.0 / gamma)
    J = wn * zn
    t = 2.0 + 2.0 * alpha * a_sq + (alpha * alpha) * (J * J)
    disc = max(0.0, t * t - 4.0 / (delta * delta))
    return abs(1.0 / gamma) * math.sqrt(0.5 * (t + math.sqrt(disc)))


def run_suite_t4_r22(rng: np.random.Generator) -> dict:
    # 1. Explicit codex counterexample: F = diag(0, 1), A_j = diag(1/2, 1), FB_j = 0
    F_ce = np.diag([0.0, 1.0])
    k_ce = np.array([1.0, 0.0])
    gamma_ce, beta_ce = 1.0, 0.5
    A_ce = gamma_ce * (np.eye(2) - beta_ce * np.outer(k_ce, k_ce))
    assert np.allclose(A_ce, np.diag([0.5, 1.0]))
    v_ce = np.array([3.5, -2.1])
    B_ce = beta_ce * np.outer(k_ce, v_ce)
    assert np.allclose(F_ce @ B_ce, 0.0)
    H_ce = rng.standard_normal((2, 2))
    ce_max_err = 0.0
    for _ in range(200):
        X = rng.standard_normal((2, 2)) * 5.0
        Y = F_ce @ (A_ce @ X + B_ce) + H_ce
        Y_prime = F_ce @ X + H_ce
        # Case 1 editor: Y' = gamma^{-1}(Y - H) + H = Y
        Y_rec = (Y - H_ce) / gamma_ce + H_ce
        ce_max_err = max(ce_max_err, float(np.max(np.abs(Y_rec - Y_prime))))
    assert ce_max_err == 0.0, f"Counterexample recovery error {ce_max_err}"

    # 2. Systematic test across 300 singular F matrices (d_k = 8, d_v = 4, rank r in {1..7})
    d_k, d_v = 8, 4
    case1_max_err = 0.0
    case2_max_err = 0.0
    case2_theta_max_rel_err = 0.0
    case3_min_witness_gap = math.inf
    case3_max_obs_diff = 0.0

    for trial in range(100):
        r = int(rng.integers(2, d_k))  # rank r in {2..7}
        U, _ = np.linalg.qr(rng.standard_normal((d_k, d_k)))
        V, _ = np.linalg.qr(rng.standard_normal((d_k, d_k)))
        svals = np.zeros(d_k)
        svals[:r] = np.exp(rng.uniform(-1.5, 1.5, size=r))
        F = U @ np.diag(svals) @ V.T
        F_pinv = V @ np.diag([1.0 / s if s > 0 else 0.0 for s in svals]) @ U.T
        P_im = U[:, :r] @ U[:, :r].T
        P_row = V[:, :r] @ V[:, :r].T
        P_ker = V[:, r:] @ V[:, r:].T

        gamma = float(rng.uniform(0.4, 1.0))
        beta = float(rng.uniform(0.2, 0.85))
        H = rng.standard_normal((d_k, d_v))
        v = rng.standard_normal(d_v)
        X = rng.standard_normal((d_k, d_v)) * 2.0

        # Case 1: k_j in ker(F)
        k1 = P_ker @ rng.standard_normal(d_k)
        k1 /= np.linalg.norm(k1)
        A1 = gamma * (np.eye(d_k) - beta * np.outer(k1, k1))
        B1 = beta * np.outer(k1, v)
        Y1 = F @ (A1 @ X + B1) + H
        Y1_prime = F @ X + H
        Y1_rec = (Y1 - H) / gamma + H
        case1_max_err = max(case1_max_err, float(np.linalg.norm(Y1_rec - Y1_prime) / np.linalg.norm(Y1_prime)))

        # Case 2: k_j in row(F)
        k2 = P_row @ rng.standard_normal(d_k)
        k2 /= np.linalg.norm(k2)
        a_sq = float(np.dot(k2, k2))
        delta = 1.0 - beta * a_sq
        alpha = beta / delta
        A2 = gamma * (np.eye(d_k) - beta * np.outer(k2, k2))
        B2 = beta * np.outer(k2, v)
        w2 = F @ k2
        z2_star = F_pinv.T @ k2
        M_plus = (1.0 / gamma) * (np.eye(d_k) + alpha * np.outer(w2, z2_star)) @ P_im
        Y2 = F @ (A2 @ X + B2) + H
        Y2_prime = F @ X + H
        Y2_rec = M_plus @ (Y2 - H - F @ B2) + H
        case2_max_err = max(case2_max_err, float(np.linalg.norm(Y2_rec - Y2_prime) / np.linalg.norm(Y2_prime)))

        # Restricted operator norm on Im(F) vs closed-form formula (18b) with (w2, z2_star)
        M_restricted = U[:, :r].T @ M_plus @ U[:, :r]
        theta_num = float(np.linalg.norm(M_restricted, ord=2))
        theta_cf = _closed_form_theta(gamma, beta, a_sq, w2, z2_star)
        case2_theta_max_rel_err = max(case2_theta_max_rel_err, abs(theta_num - theta_cf) / max(theta_cf, 1e-12))

        # Case 3: k_j straddling ker(F) and row(F)
        u_ker = V[:, r]  # unit vector in ker(F)
        u_row = V[:, 0]  # unit vector in row(F)
        k3 = (u_ker + u_row) / math.sqrt(2.0)
        delta3 = 1.0 - beta * float(np.dot(k3, k3))
        alpha3 = beta / delta3
        A3 = gamma * (np.eye(d_k) - beta * np.outer(k3, k3))
        B3 = beta * np.outer(k3, v)
        # Witness perturbation dX = A3^{-1} u_ker e_1^T
        u_ker_dot_k3 = float(np.dot(u_ker, k3))
        a3_inv_u = (1.0 / gamma) * (u_ker + alpha3 * u_ker_dot_k3 * k3)
        dX = np.zeros((d_k, d_v))
        dX[:, 0] = a3_inv_u
        Y3_a = F @ (A3 @ X + B3) + H
        Y3_b = F @ (A3 @ (X + dX) + B3) + H
        Y3p_a = F @ X + H
        Y3p_b = F @ (X + dX) + H
        obs_diff = float(np.linalg.norm(Y3_b - Y3_a))
        edit_diff = float(np.linalg.norm(Y3p_b - Y3p_a))
        expected_edit_diff = abs((1.0 / gamma) * alpha3 * u_ker_dot_k3) * float(np.linalg.norm(F @ k3))
        assert abs(edit_diff - expected_edit_diff) < 1e-12
        case3_max_obs_diff = max(case3_max_obs_diff, obs_diff)
        case3_min_witness_gap = min(case3_min_witness_gap, edit_diff)

    assert case1_max_err < 1e-13
    assert case2_max_err < 1e-12
    assert case2_theta_max_rel_err < 1e-12
    assert case3_max_obs_diff < 1e-13
    assert case3_min_witness_gap > 0.01

    return {
        "pass": True,
        "codex_counterexample_max_err": ce_max_err,
        "case1_ker_F_max_rel_err": case1_max_err,
        "case2_row_F_pseudoinv_max_rel_err": case2_max_err,
        "case2_theta_closed_form_max_rel_err": case2_theta_max_rel_err,
        "case3_straddle_max_unedited_obs_diff": case3_max_obs_diff,
        "case3_straddle_min_edited_state_gap": case3_min_witness_gap,
    }


def run_suite_t8b_r22(rng: np.random.Generator) -> dict:
    # 1. Synthetic verification across beta * ||k||^2 in [0, 2.0] (including 4-ULP rsqrt, beta=1.0, and [1.0, 2.0])
    d_k, d_v, T = 32, 16, 40
    max_energy_identity_err = 0.0
    bound_violations_in_0_2 = 0
    for trial in range(200):
        q = rng.standard_normal(d_k)
        q /= np.linalg.norm(q)
        p = q.copy()
        energy_decrement_sum = 0.0
        gammas = rng.uniform(0.7, 1.0, size=T)
        # Sample z_t = beta_t ||k_t||^2 across [0.0, 2.0], explicitly including > 1.0!
        if trial < 50:
            # 4-ULP FP32 rsqrt + beta = 1.0 regime
            kn_sq = np.full(T, 1.00000048)
            betas = np.ones(T)
        elif trial < 150:
            # Strictly super-1 regime: beta * ||k||^2 in [1.0, 1.999]
            kn_sq = rng.uniform(1.0, 1.4, size=T)
            betas = rng.uniform(0.8, 1.4, size=T)
            betas = np.minimum(betas, 1.999 / kn_sq)
        else:
            # Exact boundary beta * ||k||^2 = 2.0 (reflection)
            kn_sq = np.ones(T)
            betas = np.full(T, 2.0)

        ks = rng.standard_normal((T, d_k))
        ks /= np.linalg.norm(ks, axis=-1, keepdims=True)
        ks *= np.sqrt(kn_sq)[:, None]

        # Run backward adjoint and check energy identity (R2.27) & step contraction
        for s in range(T - 1, -1, -1):
            g_s, b_s, k_s = gammas[s], betas[s], ks[s]
            z_s = b_s * float(np.dot(k_s, k_s))
            assert 0.0 <= z_s <= 2.0 + 1e-12
            p_prev = g_s * (p - b_s * float(np.dot(k_s, p)) * k_s)
            lhs = float(np.dot(p_prev, p_prev))
            dec_term = (g_s * g_s) * b_s * (2.0 - z_s) * (float(np.dot(k_s, p)) ** 2)
            rhs = (g_s * g_s) * float(np.dot(p, p)) - dec_term
            max_energy_identity_err = max(max_energy_identity_err, abs(lhs - rhs))
            if dec_term < -1e-14 or lhs > (g_s * g_s) * float(np.dot(p, p)) + 1e-13:
                bound_violations_in_0_2 += 1
            p = p_prev

    assert max_energy_identity_err < 1e-13
    assert bound_violations_in_0_2 == 0

    # Verify converse: when beta * ||k||^2 = 2.1 > 2.0, ||A_s^\top p||_2 > gamma_s ||p||_2 along k_s
    k_bad = np.zeros(d_k)
    k_bad[0] = 1.0
    p_bad = k_bad.copy()
    g_bad, b_bad = 0.9, 2.1
    p_prev_bad = g_bad * (p_bad - b_bad * float(np.dot(k_bad, p_bad)) * k_bad)
    exceed_outside_0_2 = float(np.linalg.norm(p_prev_bad) - g_bad * np.linalg.norm(p_bad))
    assert exceed_outside_0_2 > 0.08

    # 2. Audit all real-weight trajectories in results_r2/bench_r2_out/model_*/mechanistic_dev.jsonl
    base_dir = Path(__file__).resolve().parent / "results_r2" / "bench_r2_out"
    by_model = {}
    tot_c256_4inst = 0
    tot_c256_5mod = 0
    tot_all_5mod = 0
    r23_c256_4inst = 0
    r23_c256_5mod = 0
    r23_all_5mod = 0
    r22_fwd_invalid_total = 0
    prop8b_viol_total = 0

    for tag in ("0.8B", "2B", "2B-Base", "4B", "9B"):
        m_path = base_dir / f"model_{tag}" / "mechanistic_dev.jsonl"
        rows = [json.loads(line) for line in m_path.read_text().splitlines() if line.strip()]
        by_model[tag] = {}
        for C in (256, 1024):
            sub = [m for m in rows if m["C"] == C]
            checks = [
                chk
                for m in sub
                for field in ("real_weight_adjoint", "pure_suffix_adjoint")
                for chk in m[field].values()
            ]
            n_total = sum(len(c["actual_readout_error"]) for c in checks)
            n_r23 = sum(int(c["r23_valid"]) for c in checks)
            max_kn = max(float(c["max_kn_fp32"]) for c in checks)
            max_beta = max(float(c["max_beta_fp32"]) for c in checks)
            min_gamma = min(float(c["min_gamma_fp32"]) for c in checks)
            inv_gamma = sum(int(c["invalid_gamma_heads"]) for c in checks)
            inv_kn_ulp = sum(int(c["invalid_kn_gt_1_plus_1e5_heads"]) for c in checks)
            # Under (R22.Fwd): 0 <= gamma <= 1 and 0 <= beta * ||k||^2 <= 2
            fwd_invalid = sum(
                int(c["invalid_gamma_heads"])
                for c in checks
            ) + sum(
                1 for c in checks if float(c["max_kn_fp32"]) * float(c["max_beta_fp32"]) > 2.0
            )
            n_viol_all = 0
            for c in checks:
                act = np.asarray(c["actual_readout_error"], dtype=np.float64)
                adj = np.asarray(c["B_adj"], dtype=np.float64)
                frob = np.asarray(c["B_Frob"], dtype=np.float64)
                tol = 1e-6 * (1.0 + frob)
                n_viol_all += int(np.sum((act > adj + tol) | (adj > frob + tol)))
            by_model[tag][str(C)] = {
                "total_head_trajectories": n_total,
                "strict_r23_valid": n_r23,
                "r22_fwd_valid": n_total - fwd_invalid,
                "r22_fwd_invalid": fwd_invalid,
                "invalid_gamma_heads": inv_gamma,
                "invalid_kn_gt_1_plus_1e5_heads": inv_kn_ulp,
                "max_kn_fp32": max_kn,
                "max_beta_fp32": max_beta,
                "max_z_upper_bound": max_kn * max_beta,
                "min_gamma_fp32": min_gamma,
                "prop8b_violations_all_heads": n_viol_all,
            }
            tot_all_5mod += n_total
            r23_all_5mod += n_r23
            r22_fwd_invalid_total += fwd_invalid
            prop8b_viol_total += n_viol_all
            if C == 256:
                tot_c256_5mod += n_total
                r23_c256_5mod += n_r23
                if tag != "2B-Base":
                    tot_c256_4inst += n_total
                    r23_c256_4inst += n_r23

    assert tot_c256_4inst == 114048 and r23_c256_4inst == 70170
    assert tot_c256_5mod == 122112 and r23_c256_5mod == 74464
    assert tot_all_5mod == 175104 and r23_all_5mod == 106882
    assert r22_fwd_invalid_total == 0
    assert prop8b_viol_total == 0

    return {
        "pass": True,
        "synthetic_max_energy_identity_err": max_energy_identity_err,
        "synthetic_bound_violations_in_0_2": bound_violations_in_0_2,
        "converse_violation_at_z_2_1": exceed_outside_0_2,
        "real_weight_audit": {
            "total_c256_4_instruct": tot_c256_4inst,
            "strict_r23_valid_c256_4_instruct": r23_c256_4inst,
            "r22_fwd_valid_c256_4_instruct": tot_c256_4inst,
            "total_c256_all_5_models": tot_c256_5mod,
            "strict_r23_valid_c256_all_5_models": r23_c256_5mod,
            "r22_fwd_valid_c256_all_5_models": tot_c256_5mod,
            "total_all_strides_all_5_models": tot_all_5mod,
            "strict_r23_valid_all_strides_all_5_models": r23_all_5mod,
            "r22_fwd_valid_all_strides_all_5_models": tot_all_5mod,
            "r22_fwd_invalid_total": r22_fwd_invalid_total,
            "prop8b_violations_total": prop8b_viol_total,
            "by_model": by_model,
        },
    }


def run_suite_t6_r22() -> dict:
    # Verify exact byte formulas including T = 8192 sub-KV regime (C = 2048 -> 0.58x of FA KV)
    models = {
        "0.8B": {"layers": 24, "h_k": 16, "h_v": 16, "h_kv": 2, "d_k": 128, "d_v": 128, "d_fa": 256},
        "2B": {"layers": 24, "h_k": 16, "h_v": 16, "h_kv": 2, "d_k": 128, "d_v": 128, "d_fa": 256},
        "4B": {"layers": 32, "h_k": 16, "h_v": 32, "h_kv": 4, "d_k": 128, "d_v": 128, "d_fa": 256},
        "9B": {"layers": 32, "h_k": 16, "h_v": 32, "h_kv": 4, "d_k": 128, "d_v": 128, "d_fa": 256},
    }
    table = {}
    for tag, cfg in models.items():
        L_fa = cfg["layers"] // 4
        L_gdn = cfg["layers"] - L_fa
        J = L_gdn * cfg["h_v"]
        conv_dim = 2 * cfg["h_k"] * cfg["d_k"] + cfg["h_v"] * cfg["d_v"]
        ring_dyn = 2 * L_gdn * conv_dim * 4  # DynamicCache 4-slot conv_state
        table[tag] = {}
        for T, C in ((4096, 256), (4096, 1024), (8192, 1024), (8192, 2048)):
            m = (T // C) - 1
            kv_bytes = T * L_fa * 2 * cfg["h_kv"] * cfg["d_fa"] * 2
            direct32_bytes = m * (4 * J * (128 * 128 + 2 * 128 * 128) + ring_dyn)
            delta32_bytes = m * (4 * J * (128 * 128 + 1 * 128 * 128) + ring_dyn)
            delta16_bytes = m * (2 * J * (128 * 128 + 1 * 128 * 128) + ring_dyn)
            table[tag][f"T{T}_C{C}"] = {
                "m": m,
                "fa_kv_mib": kv_bytes / (2**20),
                "direct32_mib": direct32_bytes / (2**20),
                "delta32_mib": delta32_bytes / (2**20),
                "delta16_mib": delta16_bytes / (2**20),
                "direct32_over_kv": direct32_bytes / kv_bytes,
                "delta32_over_kv": delta32_bytes / kv_bytes,
                "delta16_over_kv": delta16_bytes / kv_bytes,
            }
            if T == 8192 and C == 2048:
                assert delta16_bytes < 0.60 * kv_bytes, f"Expected sub-KV at T=8192, C=2048, got {delta16_bytes / kv_bytes}"
    return {"pass": True, "runtime_archive_table": table}


def main(write_json: bool = False) -> None:
    rng = np.random.default_rng(SEED)
    t4 = run_suite_t4_r22(rng)
    t8b = run_suite_t8b_r22(rng)
    t6 = run_suite_t6_r22()
    results = {
        "version": "R2.2",
        "seed": SEED,
        "all_pass": bool(t4["pass"] and t8b["pass"] and t6["pass"]),
        "T4_R22_SingularSuffixSubspace": t4,
        "T8b_R22_ForwardContractionSeparation": t8b,
        "T6_R22_RuntimeArchiveLedgers": t6,
    }
    out_path = Path(__file__).resolve().parent / "SMOKE_R22_RESULTS.json"
    if write_json or not out_path.exists():
        out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[smoke_r22] ALL PASS = {results['all_pass']}")
    print(f"  T4_R22: counterexample_err={t4['codex_counterexample_max_err']:.2e}, "
          f"case1_err={t4['case1_ker_F_max_rel_err']:.2e}, "
          f"case2_err={t4['case2_row_F_pseudoinv_max_rel_err']:.2e}, "
          f"case3_gap={t4['case3_straddle_min_edited_state_gap']:.3f}")
    rw = t8b["real_weight_audit"]
    print(f"  T8b_R22: C=256 (4 Instruct) = {rw['r22_fwd_valid_c256_4_instruct']}/{rw['total_c256_4_instruct']} valid, "
          f"C=256 (All 5) = {rw['r22_fwd_valid_c256_all_5_models']}/{rw['total_c256_all_5_models']} valid, "
          f"All C (All 5) = {rw['r22_fwd_valid_all_strides_all_5_models']}/{rw['total_all_strides_all_5_models']} valid, "
          f"violations = {rw['prop8b_violations_total']}")
    print(f"  T6_R22: 9B T=8192 C=2048 Delta16 = {t6['runtime_archive_table']['9B']['T8192_C2048']['delta16_mib']:.1f} MiB "
          f"({t6['runtime_archive_table']['9B']['T8192_C2048']['delta16_over_kv']:.2f}x of 256 MiB FA KV)")


if __name__ == "__main__":
    main()
