#!/usr/bin/env python3
"""Round 2.2 figures. Values are copied from BENCH_R22_REPORT.md, which is generated from
results_r22/bench_r22_out/summary_r22.json (single A100, 4 models)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent / "figures"
C = {"blue": "#0072B2", "sky": "#56B4E9", "green": "#009E73", "orange": "#E69F00",
     "verm": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 8.5,
                     "axes.labelsize": 9, "legend.fontsize": 7, "pdf.fonttype": 42})
models = ["0.8B", "2B", "4B", "9B"]
x = np.arange(4)


def fig6():
    # G2 isolation at C=256 (max relative state error)
    mono = [1.491e-4, 3.945e-5, 2.348e-4, 7.769e-3]
    proj = [8.232e-7, 1.108e-6, 7.650e-7, 3.291e-4]
    nohalo = [0.1831, 0.1826, 0.1793, 0.2334]
    fig, ax = plt.subplots(1, 2, figsize=(6.85, 2.5), gridspec_kw={"wspace": 0.3})
    w = 0.26
    a = ax[0]
    a.bar(x - w, nohalo, w, color=C["verm"], label="NoHalo")
    a.bar(x, mono, w, color=C["orange"], label=r"CHymERA-$F$ vs. monolithic oracle")
    a.bar(x + w, proj, w, color=C["green"], label=r"CHymERA-$F$ vs. proj-matched oracle")
    a.axhline(1e-3, color="k", ls="--", lw=0.8)
    a.set_yscale("log"); a.set_ylim(3e-7, 3e3)
    a.set_xticks(x); a.set_xticklabels(models)
    a.set_ylabel("Max relative state error ($C{=}256$)")
    a.set_title("(a) Seam-halo error, 4 models", fontsize=8.5, loc="left")
    a.legend(loc="upper left", fontsize=6.3)
    a.grid(True, which="both", axis="y", alpha=0.3)
    # 9B decomposition
    b = ax[1]
    labels = ["mono-\nlithic", "proj-\nmatched", "FP64\n(2 layers)", "seg-\nchunked"]
    vals = [7.769e-3, 3.291e-4, 1.557e-4, 3.291e-4]
    cols = [C["orange"], C["green"], C["blue"], C["sky"]]
    b.bar(range(4), vals, color=cols)
    for i, v in enumerate(vals):
        b.text(i, v * 1.25, f"{v:.2e}", ha="center", fontsize=6.5)
    b.axhline(1e-3, color="k", ls="--", lw=0.8)
    b.set_yscale("log"); b.set_ylim(5e-5, 4e-2)
    b.set_xticks(range(4)); b.set_xticklabels(labels, fontsize=7)
    b.set_xlabel("Reference oracle for CHymERA-$F$ (9B, $C{=}256$)", fontsize=7.5)
    b.set_ylabel("Max relative state error")
    b.set_title(r"(b) 9B, $C{=}256$: $23.6\times$ drop when matched", fontsize=8.5, loc="left")
    b.grid(True, which="both", axis="y", alpha=0.3)
    fig.savefig(FIG / "fig6_g2_isolation.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / "fig6_g2_isolation.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


CN = json.load(open(Path(__file__).resolve().parent.parent / "paper_iclr" / "cost_numbers.json"))


def fig7():
    fig, ax = plt.subplots(1, 3, figsize=(6.85, 2.5), gridspec_kw={"wspace": 0.36})
    w = 0.26
    # (a) T=8192 post-archive single-edit speedup vs SparseAnchor
    c1024 = [CN[m]["edit_sp_1024"] for m in models]
    c2048 = [CN[m]["edit_sp_2048"] for m in models]
    a = ax[0]
    a.bar(x - w / 2, c1024, w, color=C["blue"], label=r"$C{=}1024$")
    a.bar(x + w / 2, c2048, w, color=C["sky"], label=r"$C{=}2048$ ($\Delta$16)")
    for i in range(4):
        a.text(x[i] - w / 2, c1024[i] + 0.1, f"{c1024[i]:.1f}", ha="center", fontsize=6.3)
        a.text(x[i] + w / 2, c2048[i] + 0.1, f"{c2048[i]:.1f}", ha="center", fontsize=6.3)
    a.axhline(1, color=C["grey"], ls="--", lw=0.8)
    a.set_xticks(x); a.set_xticklabels(models); a.set_ylim(0, 7.6)
    a.set_ylabel("Speedup vs. SparseAnchor")
    a.set_title(r"(a) $T{=}8192$ ($n{=}4$)", fontsize=8.5, loc="left")
    a.legend(loc="upper left", fontsize=6.3)
    # (b) K=4 symmetric total speedups (build charged)
    sp256 = [CN[m]["br_sp_256"] for m in models]
    sp1024 = [CN[m]["br_sp_1024"] for m in models]
    cp256 = [CN[m]["br_cp_256"] for m in models]
    cp1024 = [CN[m]["br_cp_1024"] for m in models]
    b = ax[1]
    ww = 0.2
    b.bar(x - 1.5 * ww, sp256, ww, color=C["blue"], label=r"vs. Sparse, $C{=}256$")
    b.bar(x - 0.5 * ww, sp1024, ww, color=C["sky"], label=r"vs. Sparse, $C{=}1024$")
    b.bar(x + 0.5 * ww, cp256, ww, color=C["verm"], label=r"vs. Compact, $C{=}256$")
    b.bar(x + 1.5 * ww, cp1024, ww, color=C["orange"], label=r"vs. Compact, $C{=}1024$")
    b.axhline(1, color="k", ls="--", lw=0.8)
    b.set_xticks(x); b.set_xticklabels(models); b.set_ylim(0, 2.6)
    b.set_ylabel(r"$K{=}4$ total speedup")
    b.set_title(r"(b) Branching, $T{=}4096$", fontsize=8.5, loc="left")
    b.legend(loc="upper left", fontsize=5.8, ncol=1)
    # (c) additional archive memory / FA KV at T=8192 C=2048
    c = ax[2]
    d32 = [110.53 / 96, 110.53 / 96, 292.5 / 256, 292.5 / 256]
    d16 = [CN[m]["mem8k2048"] for m in models]
    c.bar(x - w / 2, d32, w, color=C["purple"], label=r"$\Delta$32")
    c.bar(x + w / 2, d16, w, color=C["blue"], label=r"$\Delta$16")
    for i in range(4):
        c.text(x[i] + w / 2, d16[i] + 0.03, f"{d16[i]:.2f}", ha="center", fontsize=6.3)
        c.text(x[i] - w / 2, d32[i] + 0.03, f"{d32[i]:.2f}", ha="center", fontsize=6.3)
    c.axhline(1, color="k", ls="--", lw=0.8)
    c.set_xticks(x); c.set_xticklabels(models); c.set_ylim(0, 1.5)
    c.set_ylabel("Extra archive / retained FA KV")
    c.set_title(r"(c) $T{=}8192$, $C{=}2048$", fontsize=8.5, loc="left")
    c.legend(loc="upper right", fontsize=6.3)
    for a_ in ax:
        a_.grid(True, axis="y", alpha=0.3)
    fig.savefig(FIG / "fig7_runtime_systems.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / "fig7_runtime_systems.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    fig6(); fig7(); print("ok")
