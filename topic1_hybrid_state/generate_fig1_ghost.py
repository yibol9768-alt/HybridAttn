#!/usr/bin/env python3
"""Figure 1 (teaser): a real Qwen3.5-9B example of the ghost, and its frequency by scale."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
C = dict(blue="#0072B2", red="#D55E00", green="#009E73", grey="#888888", ink="#222222", orange="#E69F00")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7.5})
MONO = {"family": "DejaVu Sans Mono", "size": 6.4}

D9 = ROOT / "results_r22" / "bench_r22_out" / "model_9B" / "module4_taskd_multislot.jsonl"
rows = [json.loads(l) for l in open(D9)]
ex = [r for r in rows if r["instance_id"] == 504][0]["methods"]
fmt = lambda s: " | ".join(s.split(" | ")[:2])
oracle = fmt(ex["Sparse-Anchor-SuffixReplay"]["pred"])
kvonly = fmt(ex["KV-Only-Evict"]["pred"])
fast = fmt(ex["CHymERA-Fast-Delta32"]["pred"])

# leak rates, heldout n=12 (BENCH_R22_REPORT Table 7.2)
leak_kv = [16.67, 83.33, 33.33, 100.0]
leak_ch = [8.33, 0.0, 0.0, 0.0]

fig = plt.figure(figsize=(5.5, 3.1))
ax = fig.add_axes([0.0, 0.0, 0.64, 1.0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.text(0.0, 9.55, "(a)", fontweight="bold", fontsize=8.5)
ax.text(0.55, 9.55, "A prompt, an edit, and what Qwen3.5-9B answers", fontsize=7.4)

def line(y, txt, color=C["ink"], bg=None, strike=False):
    if bg:
        ax.add_patch(Rectangle((0.15, y - 0.28), 9.7, 0.62, fc=bg, ec="none", zorder=0))
    ax.text(0.25, y, txt, fontdict=MONO, color=color, va="center", zorder=2)
    if strike:
        ax.plot([0.25, 9.6], [y, y], color=C["red"], lw=0.8, zorder=3)

ax.add_patch(FancyBboxPatch((0.05, 3.3), 9.9, 5.8, boxstyle="round,pad=0,rounding_size=0.15", fc="#FAFAFA", ec=C["grey"], lw=0.6))
line(8.55, "[ALPHA] GATEWAY=amber-3828-beacon | ROUTER=quartz-4486-beacon")
line(7.85, "[BETA]  GATEWAY=quartz-6460-beacon | ROUTER=jasper-7118-beacon")
line(7.15, "[ACTIVE] Current active profile = ALPHA")
line(6.45, "... unrelated log lines ...", color=C["grey"])
line(5.75, "[OVERRIDE] Switch active profile to BETA", color=C["red"], bg="#FADBD8", strike=True)
line(5.05, "[OVERRIDE] Switch active profile to BETA (x12)", color=C["red"], bg="#FADBD8", strike=True)
ax.text(9.8, 6.35, "deleted by the edit", fontsize=7.2, color=C["red"], ha="right", va="center", fontweight="bold", zorder=4)
line(4.35, "... unrelated log lines ...", color=C["grey"])
line(3.65, "Q: output the active 4-field state summary.")

ax.text(0.25, 2.65, "Exact re-run of the edited prompt (CHymERA: same answer)", fontsize=7.2, color=C["green"], va="center", fontweight="bold")
ax.text(0.25, 2.05, oracle, fontdict=MONO, color=C["green"], va="center")
ax.text(0.25, 1.35, "KV-only eviction (answers with the deleted BETA profile)", fontsize=7.2, color=C["red"], va="center", fontweight="bold")
ax.text(0.25, 0.75, kvonly, fontdict=MONO, color=C["red"], va="center")

axb = fig.add_axes([0.745, 0.19, 0.245, 0.62])
x = np.arange(4); w = 0.36
axb.bar(x - w / 2, leak_kv, w, color=C["red"], label="KV-only eviction")
axb.bar(x + w / 2, leak_ch, w, color=C["blue"], label="CHymERA")
for i, v in enumerate(leak_kv):
    axb.text(x[i] - w / 2, v + 2, f"{v:.0f}", ha="center", fontsize=7)
axb.set_xticks(x); axb.set_xticklabels(["0.8B", "2B", "4B", "9B"], fontsize=7.4)
axb.set_ylim(0, 128); axb.set_yticks([0, 25, 50, 75, 100]); axb.tick_params(axis="y", labelsize=7)
axb.set_ylabel("answers using the deleted profile (%)", fontsize=7)
axb.legend(fontsize=7.0, loc="upper left", frameon=False, borderpad=0.1, handlelength=1.0)
axb.grid(True, axis="y", alpha=0.3, lw=0.4)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)
fig.text(0.745, 0.88, "(b)", fontweight="bold", fontsize=8.5)
fig.text(0.80, 0.88, "Held-out, by model size", fontsize=7.4)
fig.savefig(OUT / "fig1_ghost.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT / "fig1_ghost.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
print(oracle, "|", kvonly, "|", fast)
