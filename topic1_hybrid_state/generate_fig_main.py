#!/usr/bin/env python3
"""Main teaser figure (Figure 1). Numbers read from results_r2 / results_r22 JSON."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle

ROOT = Path(__file__).resolve().parent.parent
R2 = ROOT / "results_r2" / "bench_r2_out"
OUT = Path(__file__).resolve().parent / "figures"
C = dict(blue="#0072B2", sky="#56B4E9", green="#009E73", orange="#E69F00", red="#D55E00",
         purple="#CC79A7", grey="#888888", ink="#222222")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7.5})


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def kl_vals():
    out = {}
    insts = [r for r in jl(R2 / "model_9B" / "instances.jsonl") if str(r["C"]) == "256" and r["split"] == "heldout_test"]
    for k in ["KV-Only-Evict", "CHymERA-Fast"]:
        out[k] = float(np.mean([r["methods"][k]["KL_gapped"] for r in insts]))
    mech = [r for r in jl(R2 / "model_9B" / "mechanistic_dev.jsonl")
            if str(r["C"]) == "256" and "Terminal-Inverse" in r.get("ablations", {})]
    out["Terminal-Inverse"] = float(np.mean([r["ablations"]["Terminal-Inverse"]["KL_gapped"] for r in mech]))
    return out


def box(ax, x, y, w, h, fc, ec, lw=0.8, r=0.04, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def main():
    kl = kl_vals()
    fig = plt.figure(figsize=(5.5, 4.1))
    # ---------------- panel (a): the problem and the method -----------------
    ax = fig.add_axes([0.0, 0.40, 1.0, 0.60])
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(0.05, 6.0, "(a)", fontweight="bold", fontsize=8.5)
    ax.text(0.55, 6.0, "Deleting a span from a 3:1 hybrid model (3 Gated DeltaNet layers per attention layer)", fontsize=7.6)

    # token strip with chunks
    nch = 8
    x0, cw = 2.45, 0.98
    y_strip = 4.55
    ax.text(0.05, y_strip + 0.22, "prompt (chunks of $C$ tokens)", fontsize=7.0, color=C["ink"], va="center")
    for i in range(nch):
        fc = "#EEF2F7"
        if i in (2, 3):
            fc = "#FADBD8"
        box(ax, x0 + i * cw, y_strip, cw - 0.06, 0.45, fc, C["grey"], lw=0.5, r=0.03)
        ax.text(x0 + i * cw + (cw - 0.06) / 2, y_strip + 0.22, f"chunk {i}", ha="center", va="center", fontsize=6.6)
    # deleted span marker (within chunks 2-3)
    sx0, sx1 = x0 + 2.35 * cw, x0 + 3.55 * cw
    ax.add_patch(Rectangle((sx0, y_strip - 0.06), sx1 - sx0, 0.57, fc="none", ec=C["red"], lw=1.4, zorder=4))
    ax.text(sx0 - 0.1, y_strip + 0.78, "deleted span", ha="left", fontsize=7, color=C["red"], fontweight="bold")
    # halo tokens
    hx0, hx1 = sx1, sx1 + 0.55
    ax.add_patch(Rectangle((hx0, y_strip - 0.06), hx1 - hx0, 0.57, fc=C["green"], ec="none", alpha=0.45, zorder=3))
    ax.text(hx0 + 0.0, y_strip - 0.33, "halo: $W{-}1$ kept tokens", ha="left", fontsize=7.2, color=C["green"], fontweight="bold")

    # attention lane
    y_att = 3.55
    ax.text(0.05, y_att + 0.2, "attention layers\nKV evicted, later keys\nshifted (gapped RoPE)", fontsize=7.0, va="center", linespacing=0.95, color=C["purple"])
    for i in range(nch):
        gone = i in (2, 3)
        box(ax, x0 + i * cw, y_att, cw - 0.06, 0.4, "white" if gone else "#E8DAEF", C["purple"] if not gone else C["grey"],
            lw=0.7 if not gone else 0.4, r=0.03)
        if gone:
            ax.plot([x0 + i * cw + 0.1, x0 + i * cw + cw - 0.16], [y_att + 0.2, y_att + 0.2], color=C["grey"], lw=0.6, ls=":")
        ax.text(x0 + 2.5 * cw, y_att + 0.2, "evict", fontsize=7.2, ha="center", va="center", color=C["grey"], zorder=5)

    # recurrent lane: KV-only evict leaves ghost
    y_g1 = 2.65
    ax.text(0.05, y_g1 + 0.2, "recurrent layers,\nKV-only eviction:\nstate still holds the\ndeleted span (ghost)", fontsize=7.0, va="center", linespacing=0.95, color=C["red"])
    for i in range(nch):
        shade = "#F5CBA7" if i >= 2 else "#EEF2F7"
        box(ax, x0 + i * cw, y_g1, cw - 0.06, 0.4, shade, C["red"] if i >= 2 else C["grey"], lw=0.6, r=0.03)
    
    # recurrent lane: CHymERA
    y_g2 = 1.45
    ax.text(0.05, y_g2 + 0.2, "recurrent layers,\nCHymERA:\nstate corrected", fontsize=7.0, va="center", linespacing=0.95, color=C["blue"])
    for i in range(nch):
        if i < 2:
            fc, ec = "#D6EAF8", C["blue"]
        elif i in (2, 3):
            fc, ec = "#D1F2EB", C["green"]
        else:
            fc, ec = "#D6EAF8", C["blue"]
        box(ax, x0 + i * cw, y_g2, cw - 0.06, 0.4, fc, ec, lw=0.7, r=0.03)
    # anchors (archive) under chunk boundaries
    for i in range(nch + 1):
        ax.plot(x0 + i * cw - 0.03, y_g2 - 0.18, marker="^", color=C["blue"], ms=3.8, zorder=5)
    ax.text(x0 + 0.0, y_g2 - 0.52, "archive: state $S_c$ and propagator $P_c$ at every chunk boundary", fontsize=7.2, color=C["blue"], va="center")
    # arrow from anchor before span to replay
    a_x = x0 + 2 * cw - 0.03
    ax.add_patch(FancyArrowPatch((a_x, y_g2 - 0.14), (a_x + 0.05, y_g2), arrowstyle="-|>", mutation_scale=6, color=C["blue"], lw=0.8))
    # replay seam label
    ax.text((x0 + 2 * cw + hx1) / 2 + 0.12, y_g2 + 0.62, "replay seam + halo", fontsize=7.4, ha="center", color=C["green"], fontweight="bold")
    ax.annotate("", xy=(hx1 + 0.1, y_g2 + 0.55), xytext=(x0 + 2 * cw, y_g2 + 0.55),
                arrowprops=dict(arrowstyle="-", color=C["green"], lw=0.9))
    # jump arrows over clean chunks
    for i in range(4, nch):
        s = x0 + i * cw - 0.05
        ax.add_patch(FancyArrowPatch((s, y_g2 + 0.43), (s + cw - 0.02, y_g2 + 0.43), connectionstyle="arc3,rad=-0.55",
                                     arrowstyle="-|>", mutation_scale=5.5, color=C["blue"], lw=0.8))
    ax.text(x0 + 6.0 * cw, y_g2 + 0.88, r"jump: $S'_{c+1}=S_{c+1}+P_c\,(S'_c-S_c)$", fontsize=7.4, ha="center", color=C["blue"])

    # ---------------- panel (b): conditioning ellipse -----------------
    axb = fig.add_axes([0.02, 0.0, 0.30, 0.37])
    axb.set_xlim(-2.2, 2.2); axb.set_ylim(-1.9, 1.7); axb.axis("off")
    axb.text(-2.2, 1.55, "(b)", fontweight="bold", fontsize=8.5)
    axb.text(-1.6, 1.55, "Why not invert the state?", fontsize=7.2)
    axb.add_patch(Ellipse((-1.05, 0.0), 1.1, 1.1, fc="#EEF2F7", ec=C["ink"], lw=0.8))
    axb.add_patch(Ellipse((1.1, 0.0), 2.0, 0.38, angle=0, fc="#FADBD8", ec=C["red"], lw=0.8))
    axb.add_patch(FancyArrowPatch((-0.4, 0.0), (0.0, 0.0), arrowstyle="-|>", mutation_scale=7, color=C["ink"], lw=0.8))
    axb.text(-0.2, 0.17, r"$\Phi$", ha="center", fontsize=7.5)
    axb.text(-1.05, -0.72, "error ball", ha="center", fontsize=7.0)
    axb.text(1.1, 0.42, r"survives: $\sigma_i$", ha="center", fontsize=7.0, color=C["red"])
    axb.text(0.0, -1.35, r"inverse amplifies by $1/\sigma_i$" + "\n" + r"($\Theta$ up to $10^{38}$)", ha="center", fontsize=7.0, color=C["red"], linespacing=1.0)

    # ---------------- panel (c): KL bars -----------------
    axc = fig.add_axes([0.40, 0.09, 0.25, 0.26])
    names = ["KV-only\nevict", "Terminal\ninverse", "CHymERA"]
    vals = [kl["KV-Only-Evict"], kl["Terminal-Inverse"], kl["CHymERA-Fast"]]
    cols = [C["orange"], C["red"], C["blue"]]
    axc.bar(range(3), vals, color=cols, width=0.62)
    axc.set_yscale("log"); axc.set_ylim(3e-4, 120)
    for i, v in enumerate(vals):
        axc.text(i, v * 1.4, f"{v:.1e}" if v < 1 else f"{v:.1f}", ha="center", fontsize=7.0)
    axc.set_xticks(range(3)); axc.set_xticklabels(names, fontsize=7.0)
    axc.tick_params(axis="y", labelsize=6.2)
    axc.set_ylabel("KL to exact answer (nats)", fontsize=7.2)
    axc.grid(True, axis="y", alpha=0.3, lw=0.4)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    fig.text(0.40, 0.375, "(c)", fontweight="bold", fontsize=8.5)
    fig.text(0.445, 0.375, "Qwen3.5-9B, held-out", fontsize=7.2)

    # ---------------- panel (d): speedups -----------------
    axd = fig.add_axes([0.76, 0.09, 0.22, 0.26])
    labs = ["edit\n(8K)", "edit\n($<$KV mem.)", "4 branches\n(build in)"]
    CN = json.load(open(Path(__file__).resolve().parent.parent / "paper_iclr" / "cost_numbers.json"))
    sv = [CN["9B"]["edit_sp_1024"], CN["9B"]["edit_sp_2048"], CN["9B"]["br_sp_1024"]]
    axd.bar(range(3), sv, color=[C["blue"], C["sky"], C["green"]], width=0.62)
    axd.axhline(1.0, color=C["grey"], ls="--", lw=0.8)
    for i, v in enumerate(sv):
        axd.text(i, v + 0.15, f"{v:.1f}$\\times$", ha="center", fontsize=7.2)
    axd.set_xticks(range(3)); axd.set_xticklabels(labs, fontsize=6.6)
    axd.set_ylim(0, 7.6); axd.tick_params(axis="y", labelsize=6.2)
    axd.set_ylabel("speedup vs. suffix replay", fontsize=7.2)
    axd.grid(True, axis="y", alpha=0.3, lw=0.4)
    for s in ("top", "right"):
        axd.spines[s].set_visible(False)
    fig.text(0.74, 0.375, "(d)", fontweight="bold", fontsize=8.5)
    fig.text(0.785, 0.375, "Qwen3.5-9B, A100", fontsize=7.2)

    fig.savefig(OUT / "fig_main.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "fig_main.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    print(kl)


if __name__ == "__main__":
    main()
