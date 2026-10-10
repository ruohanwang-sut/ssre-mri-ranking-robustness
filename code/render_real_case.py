"""Render the real-case SSRE comparison figure (standalone), v2 manual layout.

Input : reports/q2_ssre_001_qualitative_reexport_001.npz (+ .json)
Output: docs/paper/figures/ssre_real_case_001/
        ssre_real_case_001.{png,pdf,svg} + source_data.csv + caption.txt

Spec (owner 2026-10-08):
  - case file_brain_AXT2_209_2090204, slice idx 7 (0-based), condition C1 (R=4)
  - columns: VN / PMR / PMRP; rows: rho=0.25, rho=1.00; reference shown
  - one gray window for ALL image panels, set by the reference [0, max(ref)]
  - one ROI for every panel: x=136, y=136, w=h=48 (same position/magnification)
  - per-recon TRUE absolute error map abs(recon-ref)/max(ref)
  - all six error maps share ONE blue colormap scale [0, max over the six]
  - no per-panel normalization, no sharpening/denoising; nearest-neighbor
    display so no content is invented by interpolation
  - quantitative panel values are THIS case's volume PSNR over all 16 slices
"""
from pathlib import Path
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NPZ = str(ROOT / "data/qualitative_regenerated.npz")
JSN = str(ROOT / "data/qualitative_reexport.json")
OUT = str(ROOT / "outputs/qualitative")

KEY0 = "file_brain_AXT2_209_2090204"
MODELS = ("VN", "PMR", "PMRP")
MODEL_NAMES = {"VN": "VarNet", "PMR": "PromptMR v1", "PMRP": "PromptMR+"}
RHOS = ((0.25, 8), (1.00, 32))
ROI = dict(x=136, y=136, w=48, h=48)
DPI = 600

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

os.makedirs(OUT, exist_ok=True)
qual = np.load(NPZ)
with open(JSN, encoding="utf-8") as f:
    rec = json.load(f)
psnr_of = {(c["condition"], c["rho"], c["model"]): c["PSNR_regenerated"]
           for c in rec["comparison_24_cells"]}

ref = qual[f"{KEY0}__C1__target"].astype(np.float64)
assert ref.shape == (320, 320)
vmax_img = float(ref.max())

errs, panels = {}, {}
for rho, _L in RHOS:
    for m in MODELS:
        im = qual[f"{KEY0}__C1__{m}__rho_{rho}"].astype(np.float64)
        assert im.shape == (320, 320) and np.all(np.isfinite(im))
        panels[(rho, m)] = im
        errs[(rho, m)] = np.abs(im - ref) / vmax_img
vmax_err = max(e.max() for e in errs.values())

rx, ry, rw, rh = ROI["x"], ROI["y"], ROI["w"], ROI["h"]

# ------------------------------------------------------------ manual layout - #
# canvas 183 x 72 mm; 3 panel rows x 9 panel columns of square panels, plus a
# narrow colorbar column on the right; group headers as figure text; the
# 16-slice-volume PSNR sits under each full frame as a small annotation.
MM = 1 / 25.4
W, H = 183, 71
s = 15.2                     # panel size, mm
gap_x, gap_y = 1.1, 5.4
left = 16.5                  # room for rotated row labels
top = 7.2                    # room for group headers
col_cbar_w = 2.2

def ax_rect(row, col):
    x = left + col * (s + gap_x)
    y_top = top + row * (s + gap_y)
    return [x / W, 1 - (y_top + s) / H, s / W, s / H]

fig = plt.figure(figsize=(W * MM, H * MM))


def show_img(ax, img, vmin, vmax, cmap):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="nearest", origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("0.5")


def add_roi(ax):
    ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False,
                           edgecolor="red", lw=0.8))


def group_center(col0):
    x0 = left + col0 * (s + gap_x)
    return (x0 + (3 * s + 2 * gap_x) / 2) / W


for cx, txt in ((0, "Full frame (320 x 320)"),
                (3, "ROI zoom (48 x 48 px, same gray window)"),
                (6, "Absolute error, shared blue scale")):
    fig.text(group_center(cx), 1 - (top - 2.6) / H, txt,
             ha="center", va="bottom", fontsize=5.8, color="0.25")

row_lab_kw = dict(ha="center", va="center", fontsize=6.2, rotation=90,
                  color="0.05")
fig.text(6.2 / W, 1 - (top + s / 2) / H, "Reference", **row_lab_kw)
for r, (rho, L) in enumerate(RHOS, start=1):
    fig.text(6.2 / W, 1 - (top + r * (s + gap_y) + s / 2) / H,
             f"$\\rho$ = {rho:.2f}  (L = {L})", **row_lab_kw)

# ---- row 0: reference ------------------------------------------------------ #
ax = fig.add_axes(ax_rect(0, 0))
show_img(ax, ref, 0, vmax_img, "gray")
add_roi(ax)
ax.set_title("Reference (fully sampled)", fontsize=6.0, pad=2)
ax.plot([20, 70], [302, 302], color="white", lw=1.0)
ax.text(45, 296, "50 px", color="white", fontsize=5.0,
        ha="center", va="bottom")

ax = fig.add_axes(ax_rect(0, 3))
show_img(ax, ref[ry:ry + rh, rx:rx + rw], 0, vmax_img, "gray")
ax.set_title("Reference ROI", fontsize=6.0, pad=2)

# gray colorbar, vertically centered on row 0
cax = fig.add_axes([(left + 9 * s + 8 * gap_x + 2.0) / W,
                    1 - (top + s) / H, col_cbar_w / W, s / H])
cb = plt.colorbar(plt.cm.ScalarMappable(
    cmap="gray", norm=plt.Normalize(0, vmax_img)), cax=cax)
cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
cb.outline.set_linewidth(0.5)

# ---- rows 1-2: reconstructions --------------------------------------------- #
for r, (rho, L) in enumerate(RHOS, start=1):
    for c, m in enumerate(MODELS):
        ax = fig.add_axes(ax_rect(r, c))
        show_img(ax, panels[(rho, m)], 0, vmax_img, "gray")
        add_roi(ax)
        ax.set_title(MODEL_NAMES[m], fontsize=5.8, pad=2)
        px0 = ax_rect(r, c)[0] * W
        fig.text((px0 + s / 2) / W,
                 1 - (top + r * (s + gap_y) + s + 0.8) / H,
                 f"PSNR {psnr_of[('C1', rho, m)]:.2f} dB",
                 fontsize=4.9, color="0.15", ha="center", va="top")

        ax = fig.add_axes(ax_rect(r, 3 + c))
        show_img(ax, panels[(rho, m)][ry:ry + rh, rx:rx + rw], 0, vmax_img, "gray")

        ax = fig.add_axes(ax_rect(r, 6 + c))
        show_img(ax, errs[(rho, m)], 0, vmax_err, "Blues")
        ax.set_title(f"max {errs[(rho, m)].max():.4f}", fontsize=5.0, pad=2)

# blue error colorbar spanning rows 1-2
cb_top = top + 1 * (s + gap_y)
cb_h = 2 * s + gap_y
cax = fig.add_axes([(left + 9 * s + 8 * gap_x + 2.0) / W,
                    1 - (cb_top + cb_h) / H, col_cbar_w / W, cb_h / H])
cb = plt.colorbar(plt.cm.ScalarMappable(
    cmap="Blues", norm=plt.Normalize(0, vmax_err)), cax=cax)
cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
cb.outline.set_linewidth(0.5)
cb.set_label("|err| / max(ref)", fontsize=5.2)
fig.text(group_center(6), 1.2 / H,
         "PSNR = volume metric over all 16 slices of this case "
         "(not the displayed slice alone)",
         fontsize=4.9, color="0.35", ha="center", va="bottom")

base = os.path.join(OUT, "ssre_real_case_001")
fig.savefig(base + ".png", dpi=DPI)
fig.savefig(base + ".pdf")
fig.savefig(base + ".svg")
plt.close(fig)

# ---- source data ----------------------------------------------------------- #
with open(os.path.join(OUT, "source_data.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["condition", "rho", "model", "PSNR_vol16_dB", "NMSE_vol16",
                "SSIM_vol16", "PSNR_frozen", "dPSNR_dB", "err_panel_max_norm"])
    for c in rec["comparison_24_cells"]:
        if c["condition"] != "C1" or c["rho"] not in (0.25, 1.0):
            continue
        e = errs[(c["rho"], c["model"])]
        w.writerow([c["condition"], c["rho"], c["model"],
                    f'{c["PSNR_regenerated"]:.4f}', f'{c["NMSE_regenerated"]:.6f}',
                    f'{c["SSIM_regenerated"]:.6f}', f'{c["PSNR_frozen"]:.4f}',
                    f'{c["dPSNR"]:+.4f}', f"{e.max():.6f}"])

with open(os.path.join(OUT, "caption.txt"), "w", encoding="utf-8") as f:
    f.write(
        "Figure SSRE real case. Qualitative reconstruction comparison on a fastMRI "
        "brain AXT2 validation volume (case 209_2090204), slice 8 (0-based 7), "
        "acquisition-consistent C1 undersampling (R=4, equispaced mask, 32 low-"
        "frequency lines; the identical mask object is reused across support "
        "levels). Rows: normalized sensitivity-support rho = 0.25 (8 lines) and "
        "rho = 1.00 (32 lines). Columns: VarNet, PromptMR v1, PromptMR+. All "
        "image panels share one gray window [0, "
        f"{vmax_img:.4f}] fixed by the reference; no per-panel normalization, "
        "interpolation smoothing, sharpening or denoising was applied (nearest-"
        "neighbour display). Red boxes mark the fixed ROI (48 x 48 pixels at "
        f"x = {rx}, y = {ry}); ROI zooms use the same window and the same axes "
        "size as full frames (approx. 6.7x linear magnification). Right: true "
        "absolute error maps |reconstruction - reference| / max(reference); all "
        f"six error panels share one blue scale [0, {vmax_err:.4f}] (panel maxima "
        "annotated). PSNR values are volume-level metrics computed over ALL 16 "
        "slices of this case, not the displayed slice alone. Arrays were "
        "regenerated from the frozen pipeline configuration, restored official "
        "repositories and bit-identical checkpoints (SHA-256 verified) after the "
        "original qualitative export was lost; the regeneration reproduced all "
        "24 frozen volume metrics of this case with zero deviation (max |dPSNR| "
        "= 0.0 dB, max relative |dNMSE| = 0.0, max |dSSIM| = 0.0). Label: "
        "REGENERATED; byte-identity to the lost file is not claimed. Single-case "
        "images do not support population-level or clinical conclusions.\n")

print(f"vmax_img={vmax_img:.6f} vmax_err={vmax_err:.6f}")
print("figure written:", base + ".{png,pdf,svg} + source_data.csv + caption.txt")
