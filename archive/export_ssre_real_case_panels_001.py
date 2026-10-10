"""Export raw per-panel images for the real-case SSRE demonstration.

All arrays come from the REGENERATED export npz (bitwise-verified pipeline;
zero deviation on all 24 frozen volume cells of this case) -- nothing is
cropped from screenshots or figures.

Outputs -> docs/paper/figures/ssre_real_case_001/panels/  (+ a zip next to it)
  ref_C1_slice8.{png,tif}                    320x320 fully sampled reference
  recon_C1_rho{rho}_{model}.{png,tif}        6 true reconstructions
  err_C1_rho{rho}_{model}.{png,tif}          |recon-ref|/max(ref), shared scale
  sensmaps_C1_slice8_rho{rho}.{tif,png}      VarNet official sens-net output,
                                             14-coil stacks + contact sheets
                                             (the intervention's actual target)
  sensmaps_C1_slice8_rss_rho{rho}.{png,tif}  RSS combination per rho

Scaling: image PNGs use the ONE shared gray window [0, max(ref)]; error PNGs
share [0, max over the six]; TIFFs are raw float32 with NO scaling. Sensitivity
magnitude PNGs share one scale across the two rho (max over both).
"""
import importlib.util
import os
import zipfile

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "base", r"D:/lw3/complex_mri_audit/scripts/q2_ssre_qual_reexport_001.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

OUT = r"D:/lw3/docs/paper/figures/ssre_real_case_001/panels"
os.makedirs(OUT, exist_ok=True)

MODELS = ("VN", "PMR", "PMRP")
RHOS = (0.25, 1.00)
SLICE = 7
ROI_NOTE = ""  # unused

qual = np.load(base.NPZ_OUT)
key0 = base.CASE[:-3]
ref = qual[f"{key0}__C1__target"].astype(np.float64)
vmax_img = float(ref.max())

recons = {(rho, m): qual[f"{key0}__C1__{m}__rho_{rho}"].astype(np.float64)
          for rho in RHOS for m in MODELS}
errs = {k: np.abs(im - ref) / vmax_img for k, im in recons.items()}
vmax_err = max(e.max() for e in errs.values())


def save_gray(arr, vmin, vmax, stem):
    """8-bit PNG with the shared window + raw float32 TIFF."""
    a = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), mode="L").save(
        stem + ".png")
    Image.fromarray(arr.astype(np.float32), mode="F").save(stem + ".tif")


def save_rgb(rgb, stem):
    Image.fromarray((np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8),
                    mode="RGB").save(stem + ".png")
    Image.fromarray(rgb.astype(np.float32), mode="F").save(stem + ".tif")


# 1. reference
save_gray(ref, 0, vmax_img, os.path.join(OUT, "ref_C1_slice8"))

# 2. six reconstructions
for (rho, m), im in recons.items():
    save_gray(im, 0, vmax_img,
              os.path.join(OUT, f"recon_C1_rho{rho:.2f}_{m}"))

# 3a. six error maps (shared blue scale, same colormap as the figure)
cmap = plt.get_cmap("Blues")
for k, e in errs.items():
    rgba = cmap(np.clip(e / vmax_err, 0, 1))
    Image.fromarray((np.clip(rgba[..., :3], 0, 1) * 255.0 + 0.5).astype(np.uint8),
                    mode="RGB").save(os.path.join(
                        OUT, f"err_C1_rho{k[0]:.2f}_{k[1]}.png"))
    Image.fromarray(e.astype(np.float32), mode="F").save(os.path.join(
        OUT, f"err_C1_rho{k[0]:.2f}_{k[1]}.tif"))

# 3b. coil sensitivity maps: VarNet's official sens-net (the intervened stage)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
varnet = base.load_varnet()
varnet.to(device)
meta = base.official_metadata(os.path.join(base.DATA_DIR, base.CASE))
t_vn = base.VarNetDataTransform(mask_func=base.mask_func_for("C1"),
                                use_seed=True)
with h5py.File(os.path.join(base.DATA_DIR, base.CASE), "r") as hf:
    kspace = hf["kspace"][SLICE]
    target = hf[base.TARGET_KEY][SLICE]
vs = t_vn(kspace, None, target, meta, base.CASE, SLICE)
mks = vs.masked_kspace.unsqueeze(0).to(device)
mk = vs.mask.unsqueeze(0).to(device)
sens = {}
with torch.inference_mode():
    assert not torch.is_autocast_enabled()
    for rho, L in base.CONDITIONS["C1"]["grid"]:
        if rho not in RHOS:
            continue
        sm = varnet.sens_net(mks, mk, L)          # (1, coils, H, W) complex
        sens[rho] = sm[0].cpu().numpy()
        print(f"sens rho={rho}: shape={sens[rho].shape}", flush=True)

vmax_sens = max(float(np.abs(s).max()) for s in sens.values())
for rho, s in sens.items():
    if np.iscomplexobj(s):
        c = s
    else:
        c = s[..., 0] + 1j * s[..., 1]     # real/imag packed in last dim
    mag = np.abs(c)                        # (coils, H, W)
    n = mag.shape[0]
    for ci in range(n):                    # one single-page float TIFF per coil
        Image.fromarray(np.ascontiguousarray(mag[ci], dtype=np.float32),
                        mode="F").save(os.path.join(
            OUT, f"sensmaps_C1_slice8_rho{rho:.2f}_coil{ci + 1:02d}.tif"))
    rss = np.sqrt((mag ** 2).sum(axis=0))
    save_gray(rss, 0, max(float(np.sqrt((np.abs(x if np.iscomplexobj(x) else
                                             x[..., 0] + 1j * x[..., 1])
                                        ** 2).sum(axis=0)).max())
                          for x in sens.values()),
              os.path.join(OUT, f"sensmaps_C1_slice8_rss_rho{rho:.2f}"))
    # contact sheet: 14 coils, shared scale across rho
    ncol = 5
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.1, nrow * 1.1))
    for ax in axes.flat:
        ax.axis("off")
    for ci in range(n):
        ax = axes.flat[ci]
        ax.imshow(mag[ci], cmap="gray", vmin=0, vmax=vmax_sens,
                  interpolation="nearest")
        ax.set_title(f"coil {ci + 1}", fontsize=5, pad=1)
    fig.suptitle(f"VarNet sensitivity magnitude, rho={rho:.2f} (L={int(rho * 32)}), "
                 f"slice 8, shared scale [0, {vmax_sens:.3f}]", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"sensmaps_C1_slice8_rho{rho:.2f}.png"), dpi=300)
    plt.close(fig)

# zip everything
zip_path = os.path.join(os.path.dirname(OUT),
                        "ssre_real_case_001_panels_001.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(OUT)):
        z.write(os.path.join(OUT, fn), fn)
print(f"panels: {len(os.listdir(OUT))} files -> {zip_path}", flush=True)
print(f"vmax_img={vmax_img:.6f} vmax_err={vmax_err:.6f} vmax_sens={vmax_sens:.6f}",
      flush=True)
