"""Appendix D.1 figure: standard vs minimal system prompt for the 16 local paper models.
Model score = mean over the 7 meaningful instruction pairs of the per-pair median dH (bits).
Data: analysis_outputs/min_vs_std2.json (paper run selection, 0.9 validity rule)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("analysis_outputs/min_vs_std2.json"))
# (model id, label, family, label offset in points, horizontal alignment)
M = [("qwen/Qwen3-1.7B", "Qwen3-1.7B", "Qwen3", (-6, -12), "right"),
     ("qwen/Qwen3-4B", "Qwen3-4B", "Qwen3", (-6, 3), "right"),
     ("qwen/Qwen3-8B", "Qwen3-8B", "Qwen3", (-6, 2), "right"),
     ("qwen/Qwen3-14B", "Qwen3-14B", "Qwen3", (6, -11), "left"),
     ("qwen/Qwen3-32B", "Qwen3-32B", "Qwen3", (6, -11), "left"),
     ("qwen/Qwen3.5-4B", "Qwen3.5-4B", "Qwen3.5", (6, 4), "left"),
     ("qwen/Qwen3.5-9B", "Qwen3.5-9B", "Qwen3.5", (-6, 6), "right"),
     ("qwen/Qwen3.5-27B", "Qwen3.5-27B", "Qwen3.5", (-6, 5), "right"),
     ("qwen/Qwen3.5-35B-A3B", "Qwen3.5-35B-A3B", "Qwen3.5", (6, -11), "left"),
     ("meta-llama/Meta-Llama-3-8B-Instruct", "Llama-3-8B", "Llama-3", (-6, 4), "right"),
     ("meta-llama/Meta-Llama-3-70B-Instruct", "Llama-3-70B", "Llama-3", (6, 4), "left"),
     ("allenai/Olmo-3.1-32B-Instruct", "OLMo-3.1-32B", "OLMo-3.1", (6, -11), "left"),
     ("allenai/Olmo-3.1-32B-Instruct-SFT", "OLMo-3.1-32B-SFT", "OLMo-3.1", (-6, -4), "right"),
     ("allenai/Olmo-3.1-32B-Instruct-DPO", "OLMo-3.1-32B-DPO", "OLMo-3.1", (6, 4), "left"),
     ("google/gemma-4-26B-A4B-it", "Gemma-4-26B-A4B", "Gemma-4", (6, 3), "left"),
     ("google/gemma-4-31B-it", "Gemma-4-31B", "Gemma-4", (6, 4), "left")]
FAM = {"Qwen3": "#2563EB", "Qwen3.5": "#7C3AED", "Llama-3": "#DC2626", "OLMo-3.1": "#CA8A04", "Gemma-4": "#059669"}

plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
lim = (-0.4, 0.55)
ax.plot(lim, lim, color="#9CA3AF", ls="--", lw=1, zorder=0)
ax.axhline(0, color="#E5E7EB", lw=0.8, zorder=0); ax.axvline(0, color="#E5E7EB", lw=0.8, zorder=0)
texts = []
for mid, lab, fam, off, ha in M:
    r = d[mid]
    s = float(np.mean([x["med"] for x in r["std"]])); mi = float(np.mean([x["med"] for x in r["min"]]))
    ax.scatter(s, mi, s=58, color=FAM[fam], edgecolor="white", lw=0.6, zorder=3)
    texts.append(ax.annotate(lab, (s, mi), xytext=off, textcoords="offset points", fontsize=9, color="#374151", ha=ha))
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Standard prompt: mean over pairs of median $\\Delta$Entropy (bits)")
ax.set_ylabel("Minimal prompt: mean over pairs of median $\\Delta$Entropy (bits)")
ax.text(0.53, 0.50, "equal", rotation=45, ha="right", va="bottom", color="#6B7280", fontsize=9)
ax.legend(handles=[plt.Line2D([0], [0], marker="o", ls="", color=c, ms=8, label=f) for f, c in FAM.items()],
          loc="upper left", frameon=False)
fig.canvas.draw()
rend = fig.canvas.get_renderer()
bb = [(t.get_text(), t.get_window_extent(rend)) for t in texts]
ov = [(a[0], b[0]) for i, a in enumerate(bb) for b in bb[i + 1:] if a[1].overlaps(b[1])]
print("label overlaps:", ov or "none")
for ext in ("pdf", "png"):
    fig.savefig(f"analysis_outputs/sweep_min_vs_std.{ext}", dpi=200)
