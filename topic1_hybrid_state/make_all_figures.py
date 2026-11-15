#!/usr/bin/env python3
"""Clean re-draw of all paper figures from the saved JSON results.
Sizes are the final print size (ICLR text width 5.5in); fonts >= 6.5pt at that size."""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R2 = ROOT / "results_r2" / "bench_r2_out"
R1 = ROOT / "results_r1" / "probe_r1_out"
R22 = ROOT / "results_r22" / "bench_r22_out"
CN = json.load(open(ROOT / "paper_iclr" / "cost_numbers.json"))
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

C = dict(blue="#0072B2", sky="#56B4E9", green="#009E73", orange="#E69F00", red="#D55E00",
         purple="#CC79A7", grey="#8A8A8A", ink="#222222", pink="#F5CBA7")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7.5,
                     "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.8, "axes.linewidth": 0.7})
MODELS = ["0.8B", "2B", "4B", "9B"]


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid, alpha=0.3, lw=0.4)
    ax.set_axisbelow(True)


def title(ax, tag, txt):
    ax.set_title(f"({tag}) {txt}", fontsize=7.8, loc="left", pad=3, fontweight="bold")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- Figure 2: method
def box(ax, x, y, w, h, fc, ec, lw=0.8, r=0.04):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=2))


def kl_vals():
    out = {}
    insts = [r for r in jl(R2 / "model_9B" / "instances.jsonl") if str(r["C"]) == "256" and r["split"] == "heldout_test"]
    for k in ["KV-Only-Evict", "CHymERA-Fast"]:
        out[k] = float(np.mean([r["methods"][k]["KL_gapped"] for r in insts]))
    mech = [r for r in jl(R2 / "model_9B" / "mechanistic_dev.jsonl")
            if str(r["C"]) == "256" and "Terminal-Inverse" in r.get("ablations", {})]
    out["Terminal-Inverse"] = float(np.mean([r["ablations"]["Terminal-Inverse"]["KL_gapped"] for r in mech]))
    return out


def fig_method():
    kl = kl_vals()
    fig = plt.figure(figsize=(5.5, 4.0))
    ax = fig.add_axes([0.0, 0.43, 1.0, 0.57])
    ax.set_xlim(0, 11); ax.set_ylim(0, 7.0); ax.axis("off")
    ax.text(0.0, 6.75, "(a)", fontweight="bold", fontsize=8.5)
    ax.text(0.6, 6.75, "Deleting a span from a 3:1 hybrid model", fontsize=7.8, fontweight="bold")
    nch, x0, cw = 8, 2.55, 1.0
    ys, ya, yg1, yg2 = 5.45, 4.45, 3.45, 1.55
    ax.text(0.0, ys + 0.22, "prompt", fontsize=7.2, va="center")
    for i in range(nch):
        box(ax, x0 + i * cw, ys, cw - 0.07, 0.45, "#FADBD8" if i in (2, 3) else "#EEF2F7", C["grey"], lw=0.5, r=0.03)
        ax.text(x0 + i * cw + (cw - 0.07) / 2, ys + 0.22, f"chunk {i}", ha="center", va="center", fontsize=6.5)
    sx0, sx1 = x0 + 2.3 * cw, x0 + 3.55 * cw
    ax.add_patch(Rectangle((sx0, ys - 0.06), sx1 - sx0, 0.57, fc="none", ec=C["red"], lw=1.4, zorder=4))
    ax.text(sx0, ys + 0.85, "deleted span", fontsize=7.2, color=C["red"], fontweight="bold")
    ax.add_patch(Rectangle((sx1, ys - 0.06), 0.5, 0.57, fc=C["green"], ec="none", alpha=0.45, zorder=3))
    ax.text(sx1 + 0.05, ys - 0.38, "halo: $W{-}1$ kept tokens", fontsize=6.9, color=C["green"], fontweight="bold")

    ax.text(0.0, ya + 0.2, "attention\nlayers", fontsize=7.2, va="center", color=C["purple"], linespacing=1.0)
    for i in range(nch):
        gone = i in (2, 3)
        box(ax, x0 + i * cw, ya, cw - 0.07, 0.4, "white" if gone else "#E8DAEF", C["grey"] if gone else C["purple"], lw=0.5 if gone else 0.7, r=0.03)
    ax.text(x0 + 2.5 * cw, ya + 0.2, "evicted", fontsize=6.5, ha="center", va="center", color=C["grey"], zorder=5)
    ax.text(x0 + 5.0 * cw, ya - 0.34, "later keys shift by the gap", fontsize=6.7, ha="center", color=C["purple"])

    ax.text(0.0, yg1 + 0.2, "recurrent layers\nafter KV-only\neviction", fontsize=7.2, va="center", color=C["red"], linespacing=1.0)
    for i in range(nch):
        box(ax, x0 + i * cw, yg1, cw - 0.07, 0.4, C["pink"] if i >= 2 else "#EEF2F7", C["red"] if i >= 2 else C["grey"], lw=0.6, r=0.03)
    ax.text(x0 + 5.0 * cw, yg1 - 0.34, "state still holds the deleted span (the ghost)", fontsize=6.9, ha="center", color=C["red"], fontweight="bold")

    ax.text(0.0, yg2 + 0.2, "recurrent layers\nwith CHymERA", fontsize=7.2, va="center", color=C["blue"], linespacing=1.0)
    for i in range(nch):
        fc, ec = ("#D1F2EB", C["green"]) if i in (2, 3) else ("#D6EAF8", C["blue"])
        box(ax, x0 + i * cw, yg2, cw - 0.07, 0.4, fc, ec, lw=0.7, r=0.03)
    for i in range(nch + 1):
        ax.plot(x0 + i * cw - 0.035, yg2 - 0.2, marker="^", color=C["blue"], ms=4, zorder=5)
    ax.text(x0 - 0.1, yg2 - 0.58, "stored at every boundary: state $S_c$ and propagator $P_c$", fontsize=6.9, color=C["blue"], va="center")
    ax.annotate("", xy=(hx := sx1 + 0.55, yg2 + 0.62), xytext=(x0 + 2 * cw, yg2 + 0.62), arrowprops=dict(arrowstyle="-", color=C["green"], lw=1.0))
    ax.text((x0 + 2 * cw + hx) / 2, yg2 + 0.74, "replay seam + halo", fontsize=6.9, ha="center", color=C["green"], fontweight="bold")
    for i in range(4, nch):
        s = x0 + i * cw - 0.05
        ax.add_patch(FancyArrowPatch((s, yg2 + 0.43), (s + cw - 0.02, yg2 + 0.43), connectionstyle="arc3,rad=-0.55",
                                     arrowstyle="-|>", mutation_scale=5.5, color=C["blue"], lw=0.8))
    ax.text(x0 + 6.0 * cw, yg2 + 0.98, r"jump: $S^{\prime}_{c+1}=S_{c+1}+P_c\,(S^{\prime}_{c}-S_{c})$", fontsize=6.9, ha="center", color=C["blue"])

    # (b) ellipse
    axb = fig.add_axes([0.02, 0.0, 0.30, 0.37]); axb.set_xlim(-2.2, 2.2); axb.set_ylim(-2.0, 1.8); axb.axis("off")
    axb.text(-2.2, 1.6, "(b)", fontweight="bold", fontsize=8.5)
    axb.text(-1.65, 1.6, "Why not invert?", fontsize=7.8, fontweight="bold")
    axb.add_patch(Ellipse((-1.05, 0.2), 1.1, 1.1, fc="#EEF2F7", ec=C["ink"], lw=0.8))
    axb.add_patch(Ellipse((1.1, 0.2), 2.0, 0.38, fc="#FADBD8", ec=C["red"], lw=0.8))
    axb.add_patch(FancyArrowPatch((-0.4, 0.2), (0.0, 0.2), arrowstyle="-|>", mutation_scale=7, color=C["ink"], lw=0.8))
    axb.text(-0.2, 0.38, r"$\Phi$", ha="center", fontsize=8)
    axb.text(-1.05, -0.62, "error ball", ha="center", fontsize=6.9)
    axb.text(1.1, 0.62, r"memory survives: $\sigma_i$", ha="center", fontsize=6.9, color=C["red"])
    axb.text(0.0, -1.35, r"inverse amplifies by $1/\sigma_i$" + "\n" + r"(up to $10^{38}$ on real heads)", ha="center", fontsize=6.9, color=C["red"], linespacing=1.1)

    # (c) KL
    axc = fig.add_axes([0.42, 0.10, 0.25, 0.25])
    vals = [kl["KV-Only-Evict"], kl["Terminal-Inverse"], kl["CHymERA-Fast"]]
    axc.bar(range(3), vals, color=[C["orange"], C["red"], C["blue"]], width=0.62)
    axc.set_yscale("log"); axc.set_ylim(3e-4, 150)
    for i, v in enumerate(vals):
        axc.text(i, v * 1.5, f"{v:.1e}" if v < 1 else f"{v:.1f}", ha="center", fontsize=6.6)
    axc.set_xticks(range(3)); axc.set_xticklabels(["KV-only\nevict", "terminal\ninverse", "CHymERA"], fontsize=6.7)
    axc.set_ylabel("KL to exact answer", fontsize=7)
    clean(axc)
    fig.text(0.38, 0.385, "(c)", fontweight="bold", fontsize=8.5)
    fig.text(0.425, 0.385, "closeness to exact (9B)", fontsize=7.8, fontweight="bold")

    # (d) speed
    axd = fig.add_axes([0.78, 0.10, 0.21, 0.25])
    sv = [CN["9B"]["edit_sp_1024"], CN["9B"]["edit_sp_2048"], CN["9B"]["br_sp_1024"]]
    axd.bar(range(3), sv, color=[C["blue"], C["sky"], C["green"]], width=0.62)
    axd.axhline(1.0, color=C["grey"], ls="--", lw=0.8)
    for i, v in enumerate(sv):
        axd.text(i, v + 0.15, f"{v:.1f}$\\times$", ha="center", fontsize=6.9)
    axd.set_xticks(range(3)); axd.set_xticklabels(["edit\n$C$=1024", "edit\n$C$=2048", "4 branches\n$C$=1024"], fontsize=6.3)
    axd.set_ylim(0, 7.6); axd.set_ylabel("speedup over replay", fontsize=7)
    clean(axd)
    fig.text(0.74, 0.385, "(d)", fontweight="bold", fontsize=8.5)
    fig.text(0.785, 0.385, "speed (9B, 8K)", fontsize=7.8, fontweight="bold")
    save(fig, "fig_main")


# ---------------------------------------------------------------- Figure 3: oblique conditioning
def fig_oblique():
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.35), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.38})
    for m_name, col, mk in [("0.8B", C["blue"], "o"), ("2B", C["red"], "^")]:
        xs, ys = [], []
        for row in jl(R1 / f"model_{m_name}" / "mechanism.jsonl"):
            for w in row["P2"]:
                if w.get("status") == "OK":
                    xs.append(float(w["directional_ghost_survival"]))
                    ys.append(min(math.log10(max(float(w["theta_over_single_step"]), 1.0)), 30.0))
        a.scatter(xs, ys, s=5, c=col, marker=mk, alpha=0.25, edgecolors="none", label=f"Qwen3.5-{m_name}-Base")
    a.add_patch(Rectangle((0.25, 1.0), 0.78, 29.5, fc="#FFF2CC", alpha=0.35, zorder=0))
    a.axvline(0.25, color=C["ink"], ls="--", lw=0.7); a.axhline(1.0, color=C["ink"], ls="--", lw=0.7)
    a.text(0.62, 25.0, "high survival and\nhigh amplification:\n43.5% (0.8B), 46.9% (2B)", ha="center", va="center", fontsize=6.8,
           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C["orange"], lw=0.7))
    a.set_xlim(0, 1.02); a.set_ylim(-0.5, 30.5)
    a.set_xlabel("directional survival of the deleted memory")
    a.set_ylabel(r"$\log_{10}$ amplification of the inverse")
    clean(a, "both"); title(a, "a", "real heads")

    models5 = ["0.8B", "2B", "2B-Base", "4B", "9B"]
    tk, fk, cl = [], [], []
    for m in models5:
        mech = [r for r in jl(R2 / f"model_{m}" / "mechanistic_dev.jsonl")
                if str(r["C"]) == "256" and "Terminal-Inverse" in r.get("ablations", {})]
        tk.append(np.mean([r["ablations"]["Terminal-Inverse"]["KL_gapped"] for r in mech]))
        cl.append(np.sum([r["terminal_inv_clamps"] for r in mech]) / 1e6)
        ins = [r for r in jl(R2 / f"model_{m}" / "instances.jsonl") if str(r["C"]) == "256" and r["split"] == "dev"]
        fk.append(np.mean([r["methods"]["CHymERA-Fast"]["KL_gapped"] for r in ins]))
    x = np.arange(5); w = 0.36
    b.bar(x - w / 2, tk, w, color=C["red"], label="terminal-only inverse")
    b.bar(x + w / 2, fk, w, color=C["blue"], label="CHymERA")
    for i in range(5):
        b.text(x[i] - w / 2, tk[i] * 1.5, f"{cl[i]:.1f}M", ha="center", fontsize=6.3, color=C["red"])
    b.set_yscale("log"); b.set_ylim(5e-5, 3e3)
    b.set_xticks(x); b.set_xticklabels(["0.8B", "2B", "2B\nBase", "4B", "9B"], fontsize=6.8)
    b.set_ylabel("mean KL to exact answer")
    b.legend(loc="upper center", ncol=1, frameon=False, fontsize=6.6, borderpad=0.1)
    clean(b); title(b, "b", "inverse vs. replay")
    save(fig, "fig2_oblique_conditioning_and_blowup")


# ---------------------------------------------------------------- Figure 4: cost
def fig_cost():
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 2.5), gridspec_kw={"wspace": 0.55})
    x = np.arange(4); w = 0.34
    a = ax[0]
    c1 = [CN[m]["edit_sp_1024"] for m in MODELS]; c2 = [CN[m]["edit_sp_2048"] for m in MODELS]
    a.bar(x - w / 2, c1, w, color=C["blue"], label="$C$=1024")
    a.bar(x + w / 2, c2, w, color=C["sky"], label="$C$=2048")
    for i in range(4):
        a.text(x[i] - w / 2, c1[i] + 0.12, f"{c1[i]:.1f}", ha="center", fontsize=6.4)
        a.text(x[i] + w / 2, c2[i] + 0.12, f"{c2[i]:.1f}", ha="center", fontsize=6.4)
    a.axhline(1, color=C["grey"], ls="--", lw=0.8)
    a.set_xticks(x); a.set_xticklabels(MODELS); a.set_ylim(0, 8.2)
    a.set_ylabel("speedup over suffix replay")
    a.legend(loc="upper left", frameon=False, fontsize=6.6, borderpad=0.1)
    clean(a); title(a, "a", "edit, 8K tokens")
    b = ax[1]; ww = 0.2
    for off, key, col, lab in [(-1.5, "br_sp_256", C["blue"], "replay, $C$=256"), (-0.5, "br_sp_1024", C["sky"], "replay, $C$=1024"),
                               (0.5, "br_cp_256", C["red"], "re-prefill, $C$=256"), (1.5, "br_cp_1024", C["orange"], "re-prefill, $C$=1024")]:
        b.bar(x + off * ww, [CN[m][key] for m in MODELS], ww, color=col, label=lab)
    b.axhline(1, color=C["ink"], ls="--", lw=0.8)
    b.set_xticks(x); b.set_xticklabels(MODELS); b.set_ylim(0, 2.9)
    b.set_ylabel("speedup, build charged")
    b.legend(loc="upper left", frameon=False, fontsize=5.9, ncol=1, borderpad=0.1, labelspacing=0.2)
    clean(b); title(b, "b", "four branches, 4K")
    c = ax[2]
    d32 = [1.15, 1.15, 1.14, 1.14]
    d16 = [CN[m]["mem8k2048"] for m in MODELS]
    c.bar(x - w / 2, d32, w, color=C["purple"], label="Delta32")
    c.bar(x + w / 2, d16, w, color=C["blue"], label="Delta16")
    for i in range(4):
        c.text(x[i] - w / 2, d32[i] + 0.03, f"{d32[i]:.2f}", ha="center", fontsize=6.2)
        c.text(x[i] + w / 2, d16[i] + 0.03, f"{d16[i]:.2f}", ha="center", fontsize=6.2)
    c.axhline(1, color=C["ink"], ls="--", lw=0.8)
    c.text(3.45, 1.03, "KV size", fontsize=6.3, ha="right", va="bottom")
    c.set_xticks(x); c.set_xticklabels(MODELS); c.set_ylim(0, 1.6)
    c.set_ylabel("extra archive / KV cache")
    c.legend(loc="upper left", frameon=False, fontsize=6.4, ncol=2, borderpad=0.1, columnspacing=0.8)
    clean(c); title(c, "c", "memory, 8K, $C$=2048")
    save(fig, "fig7_runtime_systems")


# ---------------------------------------------------------------- Figure 5: halo / isolation (appendix)
def fig_g2():
    mono = [1.491e-4, 3.945e-5, 2.348e-4, 7.769e-3]
    proj = [8.232e-7, 1.108e-6, 7.650e-7, 3.291e-4]
    nohalo = [0.1831, 0.1826, 0.1793, 0.2334]
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3), gridspec_kw={"wspace": 0.4, "width_ratios": [1.2, 1.0]})
    x = np.arange(4); w = 0.26
    a = ax[0]
    a.bar(x - w, nohalo, w, color=C["red"], label="no halo")
    a.bar(x, mono, w, color=C["orange"], label="vs. monolithic oracle")
    a.bar(x + w, proj, w, color=C["green"], label="vs. matched oracle")
    a.axhline(1e-3, color=C["ink"], ls="--", lw=0.8); a.text(3.5, 1.3e-3, "gate $10^{-3}$", fontsize=6.4, ha="right")
    a.set_yscale("log"); a.set_ylim(3e-7, 3e3)
    a.set_xticks(x); a.set_xticklabels(MODELS); a.set_ylabel("max relative state error")
    a.legend(loc="upper left", frameon=False, fontsize=6.3, borderpad=0.1, labelspacing=0.2)
    clean(a); title(a, "a", "$C$=256, four models")
    b = ax[1]
    labels = ["mono-\nlithic", "matched\nbatching", "FP64\n(2 layers)"]
    vals = [7.769e-3, 3.291e-4, 1.557e-4]
    b.bar(range(3), vals, color=[C["orange"], C["green"], C["blue"]], width=0.6)
    for i, v in enumerate(vals):
        b.text(i, v * 1.3, f"{v:.1e}", ha="center", fontsize=6.6)
    b.axhline(1e-3, color=C["ink"], ls="--", lw=0.8)
    b.set_yscale("log"); b.set_ylim(5e-5, 4e-2)
    b.set_xticks(range(3)); b.set_xticklabels(labels, fontsize=6.8)
    b.set_ylabel("max relative state error")
    clean(b); title(b, "b", "9B, reference")
    save(fig, "fig6_g2_isolation")


# ---------------------------------------------------------------- Appendix A1: one pass vs two, base control
def fig_a1():
    summ = json.load(open(R2 / "summary_all.json"))
    models5 = ["0.8B", "2B", "2B-Base", "4B", "9B"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.4), gridspec_kw={"wspace": 0.38})
    kf, kF, wins = [], [], []
    for m in models5:
        rows = [r for r in jl(R2 / f"model_{m}" / "instances.jsonl") if str(r["C"]) == "256"]
        wins.append((sum(1 for r in rows if r["methods"]["CHymERA-Fast"]["KL_gapped"] < r["methods"]["CHymERA-F"]["KL_gapped"]), len(rows)))
        h = [r for r in rows if r["split"] == "heldout_test"]
        kf.append(np.mean([r["methods"]["CHymERA-Fast"]["KL_gapped"] for r in h]))
        kF.append(np.mean([r["methods"]["CHymERA-F"]["KL_gapped"] for r in h]))
    x = np.arange(5); w = 0.36
    a.bar(x - w / 2, kF, w, color=C["green"], label="two-pass $F$")
    a.bar(x + w / 2, kf, w, color=C["blue"], label="one-pass Fast")
    for i in range(5):
        a.text(x[i], max(kF[i], kf[i]) * 1.5, f"{wins[i][0]}/{wins[i][1]}", ha="center", fontsize=6.4)
    a.set_yscale("log"); a.set_ylim(5e-5, 0.6)
    a.set_xticks(x); a.set_xticklabels(["0.8B", "2B", "2B\nBase", "4B", "9B"], fontsize=6.8)
    a.set_ylabel("held-out mean KL ($C$=256)")
    a.legend(loc="upper left", frameon=False, fontsize=6.4, borderpad=0.1)
    clean(a); title(a, "a", "one pass vs. two (wins)")
    meths = [("CLM-NearestAnchorStaleGDN", "256", "stale\nanchor"), ("KV-Only-Evict", "256", "KV-only"), ("CHymERA-F", "256", "two-pass\n$F$"),
             ("CHymERA-Fast", "256", "Fast\n256"), ("CHymERA-Fast", "1024", "Fast\n1024")]
    base = [summ["models"]["2B-Base"]["by_C"][c]["methods_heldout_test"][k]["KL_gapped"] for k, c, _ in meths]
    inst = [summ["models"]["2B"]["by_C"][c]["methods_heldout_test"][k]["KL_gapped"] for k, c, _ in meths]
    x = np.arange(5)
    b.bar(x - w / 2, base, w, color=C["orange"], label="2B-Base")
    b.bar(x + w / 2, inst, w, color=C["blue"], label="2B (post-trained)")
    b.set_yscale("log"); b.set_ylim(1e-4, 30)
    b.set_xticks(x); b.set_xticklabels([m[2] for m in meths], fontsize=6.4)
    b.set_ylabel("held-out mean KL")
    b.legend(loc="upper right", frameon=False, fontsize=6.4, borderpad=0.1)
    clean(b); title(b, "b", "pre- vs. post-trained control")
    save(fig, "figA1_onepass_control")


# ---------------------------------------------------------------- Appendix A2: halo per model / per layer
def fig_a2():
    summ = json.load(open(R2 / "summary_all.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.4), gridspec_kw={"wspace": 0.38, "width_ratios": [1.15, 1.0]})
    spec = [("0.8B", "256"), ("2B", "256"), ("2B-Base", "256"), ("4B", "256"), ("9B", "256"), ("9B", "1024")]
    e0, e1, e2 = [], [], []
    for m, c in spec:
        g = summ["models"][m]["by_C"][c]["gates"]["G2"]
        e0.append(g["max_layer_rel_err_nohalo"]); e1.append(g["max_layer_rel_err_naive_halo"]); e2.append(g["max_layer_rel_err_F"])
    x = np.arange(len(spec)); w = 0.27
    a.bar(x - w, e0, w, color=C["red"], label="no halo")
    a.bar(x, e1, w, color=C["orange"], label="per-span halo")
    a.bar(x + w, e2, w, color=C["green"], label="coalesced halo")
    a.axhline(1e-3, color=C["ink"], ls="--", lw=0.8)
    a.set_yscale("log"); a.set_ylim(3e-5, 40)
    a.set_xticks(x); a.set_xticklabels([f"{m}\n{c}" for m, c in spec], fontsize=6.2)
    a.set_ylabel("worst layer relative error")
    a.legend(loc="upper left", frameon=False, fontsize=6.3, ncol=1, borderpad=0.1, labelspacing=0.2)
    clean(a); title(a, "a", "short islands, vs. $F$ oracle")
    for m, c, col, ls in [("4B", "256", C["green"], "-"), ("9B", "256", C["blue"], "--"), ("9B", "1024", C["sky"], "-.")]:
        rows = [r for r in jl(R2 / f"model_{m}" / "mechanistic_dev.jsonl")
                if str(r["C"]) == c and r["task"] == "TaskB_RAGEvict" and r["split"] == "heldout_test"]
        layers = sorted(int(k) for k in rows[0]["rel_err_chymera_F_by_layer"].keys())
        med = [np.median([r["rel_err_chymera_F_by_layer"][str(l)] for r in rows]) for l in layers]
        b.plot(range(len(layers)), med, color=col, ls=ls, marker="o", ms=2.2, lw=1.1, label=f"{m}, $C$={c}")
        if m == "9B" and c == "256":
            nm = [np.median([r["rel_err_nohalo_by_layer"][str(l)] for r in rows]) for l in layers]
            b.plot(range(len(layers)), nm, color=C["red"], ls=":", marker="s", ms=2.2, lw=1.1, label="9B no halo")
    b.axhline(1e-3, color=C["ink"], ls="--", lw=0.8)
    b.set_yscale("log"); b.set_ylim(1e-9, 3)
    b.set_xlabel("recurrent layer index"); b.set_ylabel("median relative error")
    b.legend(loc="lower center", frameon=False, fontsize=6.0, borderpad=0.1, labelspacing=0.15, ncol=2, columnspacing=0.8)
    clean(b); title(b, "b", "error by depth")
    save(fig, "figA2_halo_depth")


if __name__ == "__main__":
    fig_method(); fig_oblique(); fig_cost(); fig_g2(); fig_a1(); fig_a2()
