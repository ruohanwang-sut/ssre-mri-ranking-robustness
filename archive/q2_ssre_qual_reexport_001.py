"""Q2 SSRE qualitative re-export 001 -- ONE case, frozen pipeline, no science rerun.

Purpose
-------
Regenerate the Appendix-B qualitative reconstruction arrays for the FIRST
frozen qualitative case (file_brain_AXT2_209_2090204.h5) after the original
`q2_ssre_001_qualitative.npz` was lost in the 2026-09-15 external cleanup.
This script NEVER re-runs the SSRE matrix, never writes to any frozen report,
and exists only to (a) prove, against the frozen per-volume metrics, that the
restored pipeline reproduces the science run on this volume, and (b) re-export
the display arrays (slice 7, all conditions/models/rho) for figures.

Provenance
----------
Pipeline constants, transforms, loaders (incl. the recorded PMRP n_buffer
shim), metric_set and the per-(slice, condition) single-mask reuse pattern are
copied verbatim from D:/lw3/complex_mri_audit/run_q2_ssre_001.py (science run
sha256 f6ee1fbc3ab0575daa21f7e79d32267420ca8d7028d97673de5342ae97de8057).
Checkpoints re-downloaded 2026-10-08 and verified bit-identical to the frozen
SHA-256 values below. Repos restored at the frozen commits via codeload
tarballs. fastmri_deps re-vendored with the recorded G2 command (skimage
0.25.2 vs the 0.25.0 in site-packages at the original run -- affects ONLY
fastmri.evaluate.ssin -> SSIM is reported, not gated).

Authorization
-------------
Owner-approved 2026-10-06 (one-case qualitative re-export; script-first:
this script + acceptance criteria are presented BEFORE any run). This file
must be reviewed and approved before execution. It refuses to run if either
output path already exists.

Hard gates (pre-declared; any failure -> write stop record, exit 1)
-------------------------------------------------------------------
G1 structure : kspace shape (16, 14, 768, 396); target (16, 384, 384);
               acquisition AXT2 (frozen G3 record for this file).
G2 mask      : C1 cols==123, offset==3, nlf==32; C2 cols==64, offset==3,
               nlf==16 (frozen q2_ssre_001.json rows for this volume);
               mask bitwise-equal across VN/PMRP/PMR on every slice;
               mask bitwise-equal across all 16 slices within a condition
               (single sample object per (slice, condition) reused across
               all four rho -- the frozen acquisition-consistency design).
G3 metrics   : all 24 (cond, rho, model) volume-level cells reproduce the
               frozen cells embedded below: |dPSNR| <= 0.05 dB AND
               |dNMSE| / NMSE_frozen <= 2% (GPU nondeterminism tolerance;
               pre-declared suggestion, not a community standard).
               SSIM: REPORT ONLY (skimage version drift), expected |d| <= 1e-3.
G4 purity    : C1 and C2 target crops bitwise identical (frozen property);
               all arrays finite.

Outputs
-------
reports/q2_ssre_001_qualitative_reexport_001.npz       (same key scheme as the
    frozen capture: {key}__{cond}__target and {key}__{cond}__{model}__rho_{rho},
    26 keys, float32 320x320; this file is a REGENERATION, byte-identity to the
    lost original is NOT claimed anywhere)
reports/q2_ssre_001_qualitative_reexport_001.json      (provenance + gate
    results + the 24-cell frozen-vs-regenerated comparison table)
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time

DEPS_DIR = r"D:/lw3/_external_baselines/fastmri_deps"
FASTMRI_REPO = r"D:/lw3/_external_baselines/fastMRI_official"
PMRP_REPO = r"D:/lw3/_external_baselines/PromptMR_plus_official"
for _p in (DEPS_DIR, FASTMRI_REPO, PMRP_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import h5py  # noqa: E402

import fastmri  # noqa: E402
import fastmri.data.transforms as T  # noqa: E402
from fastmri.data.subsample import create_mask_for_mask_type  # noqa: E402
from fastmri.data.transforms import VarNetDataTransform  # noqa: E402
from fastmri.evaluate import nmse, psnr, ssim  # noqa: E402
from fastmri.models.varnet import VarNet  # noqa: E402

import data.transforms as PT  # noqa: E402  (official PromptMR repo 'data' package)
from data.transforms import FastmriDataTransform  # noqa: E402
from pl_modules import PromptMrModule  # noqa: E402

# --------------------------------------------------------------------------- #
# frozen identities (run_q2_ssre_001.py, verified 2026-10-08)
# --------------------------------------------------------------------------- #
CASE = "file_brain_AXT2_209_2090204.h5"
DATA_DIR = r"D:/lw3/fastmri_brain_val/multicoil_val"
CKPT_VARNET_PATH = r"D:/lw3/_external_baselines/weights/brain_leaderboard_state_dict.pt"
CKPT_VARNET_SHA256 = "2cdfdde2fe662f8995316a7ba981d8d3f405d2a5fd82a61bfa1bc4e56062262f"
CKPT_PMR_PATH = r"D:/lw3/_external_baselines/weights/promptmr-epoch=44-step=2083275.ckpt"
CKPT_PMR_SHA256 = "67a30557acdab3c19f118ea3d0cd63cc51a32b92326f09fd37ca89156369954e"
CKPT_PMR_BYTES = 314103087
CKPT_PMRP_PATH = r"D:/Google download/promptmr-plus-epoch%3D44-step%3D1591830.ckpt"
CKPT_PMRP_SHA256 = "42722018604944c567c598ddf5c488d135793ef337359a1da03ad3d4301e177e"

TARGET_KEY = "reconstruction_rss"
EVAL_CROP = (320, 320)
EXPECTED_SLICES = 16
EXPECTED_COILS = 14
EXPECTED_KSPACE_SHAPE = (16, 14, 768, 396)   # frozen G3 record
EXPECTED_TARGET_SHAPE = (16, 384, 384)
PMRP_UNIFORM_RESOLUTION = (384, 384)
PMRP_NUM_ADJ_SLICES = 5
PMR_NUM_ADJ_SLICES = 3
NUM_CASCADES, CHANS, POOLS = 12, 18, 4
SENS_CHANS, SENS_POOLS = 8, 4
MASK_TYPE = "equispaced"
C1 = dict(center_fraction=0.08, acceleration=4, l_nom=32,
          grid=((0.25, 8), (0.50, 16), (0.75, 24), (1.00, 32)))
C2 = dict(center_fraction=0.04, acceleration=8, l_nom=16,
          grid=((0.25, 4), (0.50, 8), (0.75, 12), (1.00, 16)))
CONDITIONS = {"C1": C1, "C2": C2}
RHO_GRID = (0.25, 0.50, 0.75, 1.00)
MODELS = ("VN", "PMR", "PMRP")
SEED = 20260901
QUAL_SLICE_IDX = 7  # 0-based, frozen

# frozen mask measurements for THIS volume (q2_ssre_001.json per_condition_rows)
FROZEN_MASK = {"C1": {"cols": 123, "offset": 3, "nlf": 32},
               "C2": {"cols": 64, "offset": 3, "nlf": 16}}

# frozen metric cells (cond, rho, model) -> (PSNR, NMSE, SSIM); verbatim from
# q2_ssre_001.json per_condition_rows for this volume
FROZEN_CELLS = {
    ("C1", 0.25, "VN"): (38.68191913174342, 0.004709006681538088, 0.9537831041290928),
    ("C1", 0.50, "VN"): (39.1818972640049, 0.004196927755054286, 0.9590145574785387),
    ("C1", 0.75, "VN"): (39.34774826253404, 0.004039674692381523, 0.9601892020530735),
    ("C1", 1.00, "VN"): (39.361741653438905, 0.004026679414609835, 0.9601579207390604),
    ("C1", 0.25, "PMR"): (39.62466522339199, 0.003790135158896376, 0.9606025683039622),
    ("C1", 0.50, "PMR"): (39.784531183850206, 0.0036531550179072555, 0.9622260627229515),
    ("C1", 0.75, "PMR"): (39.80556298783426, 0.00363550646302442, 0.961930076942323),
    ("C1", 1.00, "PMR"): (39.92954295850693, 0.0035331894419005348, 0.9628124599874517),
    ("C1", 0.25, "PMRP"): (37.93055386296684, 0.005598424600788568, 0.926751730120177),
    ("C1", 0.50, "PMRP"): (39.212528001652046, 0.004167431016352818, 0.9509329426317429),
    ("C1", 0.75, "PMRP"): (39.98543462269117, 0.0034880102906831834, 0.9612568188913856),
    ("C1", 1.00, "PMRP"): (40.19444566963049, 0.0033241196903016053, 0.9636507868869802),
    ("C2", 0.25, "VN"): (35.01693645589413, 0.010950339251956863, 0.9206174645869628),
    ("C2", 0.50, "VN"): (35.8202446884734, 0.00910116003302323, 0.9341368973257119),
    ("C2", 0.75, "VN"): (36.04550501785868, 0.008641133565375358, 0.9383041260036677),
    ("C2", 1.00, "VN"): (36.13549117958924, 0.008463930759114882, 0.9390163302566414),
    ("C2", 0.25, "PMR"): (35.86733230811587, 0.009003015317025253, 0.9228786508302326),
    ("C2", 0.50, "PMR"): (36.49782223900011, 0.007786440977639028, 0.9379078301373902),
    ("C2", 0.75, "PMR"): (36.80729632352597, 0.0072508945210256765, 0.9429988837303971),
    ("C2", 1.00, "PMR"): (36.91258536672667, 0.007077219788419554, 0.9435362874891979),
    ("C2", 0.25, "PMRP"): (35.615157338536605, 0.009541254626523485, 0.910565238405881),
    ("C2", 0.50, "PMRP"): (36.72242304607848, 0.007393990959332426, 0.9377479283384688),
    ("C2", 0.75, "PMRP"): (37.14236439147208, 0.006712507449482782, 0.9447200174568341),
    ("C2", 1.00, "PMRP"): (37.28403217074203, 0.006497076834353847, 0.9460874614200281),
}

PSNR_TOL_DB = 0.05        # pre-declared GPU nondeterminism tolerance
NMSE_TOL_REL = 0.02       # pre-declared GPU nondeterminism tolerance

NPZ_OUT = r"D:/lw3/complex_mri_audit/reports/q2_ssre_001_qualitative_reexport_001.npz"
JSON_OUT = r"D:/lw3/complex_mri_audit/reports/q2_ssre_001_qualitative_reexport_001.json"


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stop(reason: str) -> int:
    rec = {"status": "STOP", "reason": reason, "case": CASE,
           "script": os.path.abspath(__file__),
           "script_sha256": sha256_of(os.path.abspath(__file__))}
    with open(JSON_OUT + ".stop", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[STOP] {reason}", flush=True)
    return 1


def mask_cols_offset(mask_bool_vec):
    return int(mask_bool_vec.sum().item()), int(mask_bool_vec.float().argmax().item())


def mask_func_for(cond_name):
    cfg = CONDITIONS[cond_name]
    return create_mask_for_mask_type(MASK_TYPE, [cfg["center_fraction"]],
                                     [cfg["acceleration"]])


def official_metadata(fname):
    """Verbatim from run_q2_ssre_001.py (locked SliceDataset._retrieve_metadata)."""
    from xml.etree import ElementTree as etree  # noqa
    from fastmri.data.mri_data import et_query
    with h5py.File(fname, "r") as hf:
        et_root = etree.fromstring(hf["ismrmrd_header"][()])
        enc = ["encoding", "encodedSpace", "matrixSize"]
        enc_size = (int(et_query(et_root, enc + ["x"])),
                    int(et_query(et_root, enc + ["y"])),
                    int(et_query(et_root, enc + ["z"])))
        rec = ["encoding", "reconSpace", "matrixSize"]
        recon_size = (int(et_query(et_root, rec + ["x"])),
                      int(et_query(et_root, rec + ["y"])),
                      int(et_query(et_root, rec + ["z"])))
        lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
        enc_limits_center = int(et_query(et_root, lims + ["center"]))
        enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1
        padding_left = enc_size[1] // 2 - enc_limits_center
        padding_right = padding_left + enc_limits_max
        return {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
            **{k: (v.item() if hasattr(v, "item") else v) for k, v in dict(hf.attrs).items()},
        }


def magnitude_from_kspace(kspace_pred):
    return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)


def varnet_forward_with_sensmaps(model, masked_kspace, mask, sens_maps):
    kspace_pred = masked_kspace.clone()
    for cascade in model.cascades:
        kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
    return kspace_pred


def varnet_fwd(model, masked_kspace, mask, num_low_frequencies):
    sens_maps = model.sens_net(masked_kspace, mask, num_low_frequencies)
    return varnet_forward_with_sensmaps(model, masked_kspace, mask, sens_maps)


def metric_set(gt_stack, pred_stack):
    """Verbatim from run_q2_ssre_001.py: fastmri.evaluate on (16,320,320)."""
    return {"NMSE": float(nmse(gt_stack, pred_stack).item()),
            "PSNR": float(psnr(gt_stack, pred_stack).item()),
            "SSIM": float(ssim(gt_stack, pred_stack).item())}


def load_varnet():
    model = VarNet(num_cascades=NUM_CASCADES, pools=POOLS, chans=CHANS,
                   sens_pools=SENS_POOLS, sens_chans=SENS_CHANS)
    sd = torch.load(CKPT_VARNET_PATH, weights_only=True, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, (missing, unexpected)
    assert all(p.dtype == torch.float32 for p in model.parameters()), "VarNet not fp32"
    model.eval()
    return model


def load_pmrp(device):
    ck = torch.load(CKPT_PMRP_PATH, weights_only=True, map_location="cpu")
    hp = dict(ck["hyper_parameters"])
    assert int(hp["num_adj_slices"]) == PMRP_NUM_ADJ_SLICES
    module = PromptMrModule(**hp)
    missing, unexpected = module.load_state_dict({k: v for k, v in ck["state_dict"].items()},
                                                 strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, (missing, unexpected)
    # recorded upstream-compat shim (the ONLY permitted shim; run_q2_ssre_001.py)
    for blk in module.promptmr.cascades:
        assert not hasattr(blk, "n_buffer")
        blk.n_buffer = int(blk.model.n_buffer)
    module.eval()
    module.to(device)
    return module


def load_pmr(device):
    ck = torch.load(CKPT_PMR_PATH, weights_only=True, map_location="cpu")
    hp = dict(ck["hyper_parameters"])
    assert hp["model_version"] == "promptmr"
    assert int(hp["num_adj_slices"]) == PMR_NUM_ADJ_SLICES
    assert bool(hp["adaptive_input"]) is False and int(hp["n_buffer"]) == 0
    module = PromptMrModule(**hp)
    missing, unexpected = module.load_state_dict({k: v for k, v in ck["state_dict"].items()},
                                                 strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, (missing, unexpected)
    for blk in module.promptmr.cascades:
        assert not hasattr(blk, "n_buffer")
    assert int(module.promptmr.n_buffer) == 0
    module.eval()
    module.to(device)
    return module


def main() -> int:
    t0 = time.perf_counter()
    for out in (NPZ_OUT, JSON_OUT):
        if os.path.exists(out):
            return stop(f"output already exists: {out} (refusing to overwrite)")

    ck_ok = (sha256_of(CKPT_VARNET_PATH) == CKPT_VARNET_SHA256
             and sha256_of(CKPT_PMR_PATH) == CKPT_PMR_SHA256
             and os.path.getsize(CKPT_PMR_PATH) == CKPT_PMR_BYTES
             and sha256_of(CKPT_PMRP_PATH) == CKPT_PMRP_SHA256)
    if not ck_ok:
        return stop("checkpoint SHA-256 mismatch against frozen values")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    varnet = load_varnet()
    varnet.to(device)
    pmrp = load_pmrp(device)
    pmr = load_pmr(device)

    path = os.path.join(DATA_DIR, CASE)
    if not os.path.exists(path):
        return stop(f"case file missing: {path}")

    # ---- G1: structure ----------------------------------------------------- #
    with h5py.File(path, "r") as hf:
        if tuple(hf["kspace"].shape) != EXPECTED_KSPACE_SHAPE:
            return stop(f"kspace shape {hf['kspace'].shape} != {EXPECTED_KSPACE_SHAPE}")
        if tuple(hf[TARGET_KEY].shape) != EXPECTED_TARGET_SHAPE:
            return stop(f"target shape {hf[TARGET_KEY].shape} != {EXPECTED_TARGET_SHAPE}")
        attrs = {k: (v.decode() if isinstance(v, bytes) else v)
                 for k, v in hf.attrs.items()}
        if str(attrs.get("acquisition")) != "AXT2":
            return stop(f"acquisition {attrs.get('acquisition')!r} != 'AXT2'")
        meta = official_metadata(path)

        all_crops = {c: {m: {r: [] for r in RHO_GRID} for m in MODELS}
                     for c in ("C1", "C2")}

        transforms_vn = {"C1": VarNetDataTransform(mask_func=mask_func_for("C1"),
                                                   use_seed=True),
                         "C2": VarNetDataTransform(mask_func=mask_func_for("C2"),
                                                   use_seed=True)}
        transforms_pm = {c: FastmriDataTransform(mask_func=mask_func_for(c),
                                                 uniform_resolution=PMRP_UNIFORM_RESOLUTION,
                                                 use_seed=True, mask_type=None,
                                                 test_num_low_frequencies=None)
                         for c in ("C1", "C2")}

        qual = {}
        targets = {"C1": [], "C2": []}
        gates = {"G1_structure": "PASS", "G2_mask": "PASS",
                 "G3_metrics": "PENDING", "G4_purity": "PASS"}
        comparison = []

        for cond_name in ("C1", "C2"):
            cfg = CONDITIONS[cond_name]
            crops = all_crops[cond_name]
            prev_cols_off = None
            for s in range(EXPECTED_SLICES):
                kspace = hf["kspace"][s]
                target = hf[TARGET_KEY][s]
                # one sample object per (slice, condition), reused across rho
                vs = transforms_vn[cond_name](kspace, None, target, meta, CASE, s)
                cols, off = mask_cols_offset(vs.mask[0, 0, :, 0])
                fm = FROZEN_MASK[cond_name]
                if int(vs.num_low_frequencies) != cfg["l_nom"]:
                    gates["G2_mask"] = f"FAIL s{s}: nlf {vs.num_low_frequencies}"
                    return stop(f"G2: {cond_name} s{s} nlf {vs.num_low_frequencies}")
                if cols != fm["cols"] or off != fm["offset"]:
                    gates["G2_mask"] = f"FAIL s{s}: cols {cols} off {off}"
                    return stop(f"G2: {cond_name} s{s} cols={cols} off={off} "
                                f"(frozen {fm})")
                if prev_cols_off is not None and (cols, off) != prev_cols_off:
                    gates["G2_mask"] = f"FAIL s{s}: mask drift across slices"
                    return stop(f"G2: {cond_name} mask drift at s{s}")
                prev_cols_off = (cols, off)

                idx5 = [min(max(s + j, 0), EXPECTED_SLICES - 1) for j in range(-2, 3)]
                stacked5 = np.concatenate([hf["kspace"][i] for i in idx5], axis=0)
                ps5 = transforms_pm[cond_name](stacked5, None, None, meta, CASE, s)
                idx3 = [min(max(s + j, 0), EXPECTED_SLICES - 1) for j in range(-1, 2)]
                stacked3 = np.concatenate([hf["kspace"][i] for i in idx3], axis=0)
                ps3 = transforms_pm[cond_name](stacked3, None, None, meta, CASE, s)

                vn_vec = vs.mask[0, 0, :, 0].cpu()
                if not (torch.equal(vn_vec, ps5.mask[0, 0, :, 0].cpu())
                        and torch.equal(vn_vec, ps3.mask[0, 0, :, 0].cpu())):
                    gates["G2_mask"] = f"FAIL s{s}: cross-model mask mismatch"
                    return stop(f"G2: cross-model mask mismatch at {cond_name} s{s}")

                vn_tcrop = T.center_crop(vs.target, EVAL_CROP).numpy().astype(np.float64)
                p5_tcrop = T.center_crop(T.to_tensor(target),
                                         EVAL_CROP).numpy().astype(np.float64)
                if not np.array_equal(vn_tcrop, p5_tcrop):
                    gates["G4_purity"] = f"FAIL s{s}: target crop mismatch"
                    return stop(f"G4: target crop mismatch at {cond_name} s{s}")
                targets[cond_name].append(vn_tcrop)

                vn_mks = vs.masked_kspace.unsqueeze(0).to(device)
                vn_mk = vs.mask.unsqueeze(0).to(device)
                p5_mks = ps5.masked_kspace.unsqueeze(0).to(device)
                p5_mk = ps5.mask.unsqueeze(0).to(device)
                p3_mks = ps3.masked_kspace.unsqueeze(0).to(device)
                p3_mk = ps3.mask.unsqueeze(0).to(device)
                with torch.inference_mode():
                    assert not torch.is_autocast_enabled()
                    for rho, L in cfg["grid"]:
                        kp = varnet_fwd(varnet, vn_mks, vn_mk, L)
                        mag = magnitude_from_kspace(kp)
                        crops["VN"][rho].append(
                            T.center_crop(mag, EVAL_CROP)[0].cpu().numpy().astype(np.float64))
                        del kp, mag
                        o5 = pmrp(p5_mks, p5_mk, torch.tensor([L], device=device),
                                  [ps5.mask_type])
                        img5 = T.center_crop(PT.center_crop(o5["img_pred"], ps5.crop_size),
                                             EVAL_CROP)
                        crops["PMRP"][rho].append(img5[0].cpu().numpy().astype(np.float64))
                        del o5, img5
                        o3 = pmr(p3_mks, p3_mk, torch.tensor([L], device=device),
                                 [ps3.mask_type])
                        img3 = T.center_crop(PT.center_crop(o3["img_pred"], ps3.crop_size),
                                             EVAL_CROP)
                        crops["PMR"][rho].append(img3[0].cpu().numpy().astype(np.float64))
                        del o3, img3
                del vn_mks, vn_mk, p5_mks, p5_mk, p3_mks, p3_mk
                print(f"[{cond_name}] slice {s + 1}/{EXPECTED_SLICES} done "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

            # capture arrays for this condition (same key scheme as frozen npz)
            key0 = CASE[:-3]
            qual[f"{key0}__{cond_name}__target"] = \
                np.stack(targets[cond_name])[QUAL_SLICE_IDX].astype(np.float32)
            for m in MODELS:
                for rho, _L in cfg["grid"]:
                    qual[f"{key0}__{cond_name}__{m}__rho_{rho}"] = \
                        np.stack(crops[m][rho])[QUAL_SLICE_IDX].astype(np.float32)

        # ---- G4: cross-condition target purity -------------------------------- #
        if not np.array_equal(targets["C1"][QUAL_SLICE_IDX], targets["C2"][QUAL_SLICE_IDX]):
            gates["G4_purity"] = "FAIL: C1/C2 target crops differ"
            return stop("G4: C1/C2 target crops differ on capture slice")

    # ---- G3: metric reproduction against the 24 frozen cells ---------------- #
    gt_stacks = {c: np.stack(targets[c]) for c in ("C1", "C2")}
    worst_psnr = worst_nmse = worst_ssim = 0.0
    for cond_name in ("C1", "C2"):
        cfg = CONDITIONS[cond_name]
        for m in MODELS:
            for rho, _L in cfg["grid"]:
                pred = np.stack(all_crops[cond_name][m][rho])
                ms = metric_set(gt_stacks[cond_name], pred)
                fpsnr, fnmse, fssim = FROZEN_CELLS[(cond_name, rho, m)]
                d_psnr = abs(ms["PSNR"] - fpsnr)
                d_nmse = abs(ms["NMSE"] - fnmse) / fnmse
                d_ssim = abs(ms["SSIM"] - fssim)
                worst_psnr = max(worst_psnr, d_psnr)
                worst_nmse = max(worst_nmse, d_nmse)
                worst_ssim = max(worst_ssim, d_ssim)
                comparison.append({"condition": cond_name, "rho": rho, "model": m,
                                   "PSNR_regenerated": ms["PSNR"], "PSNR_frozen": fpsnr,
                                   "dPSNR": ms["PSNR"] - fpsnr,
                                   "NMSE_regenerated": ms["NMSE"], "NMSE_frozen": fnmse,
                                   "dNMSE_rel": d_nmse,
                                   "SSIM_regenerated": ms["SSIM"], "SSIM_frozen": fssim,
                                   "dSSIM": ms["SSIM"] - fssim})
                if d_psnr > PSNR_TOL_DB or d_nmse > NMSE_TOL_REL:
                    gates["G3_metrics"] = (
                        f"FAIL {cond_name} rho={rho} {m}: dPSNR={d_psnr:.4f} "
                        f"dNMSE_rel={d_nmse:.4%}")
                    return stop(gates["G3_metrics"])
    gates["G3_metrics"] = "PASS"

    os.makedirs(os.path.dirname(NPZ_OUT), exist_ok=True)
    np.savez_compressed(NPZ_OUT, **qual)
    record = {
        "status": "COMPLETE",
        "case": CASE,
        "character": "regeneration of the lost q2_ssre_001_qualitative.npz content "
                     "for this case; byte-identity to the lost file NOT claimed",
        "pipeline_source": "run_q2_ssre_001.py (sha256 f6ee1fbc3ab0575daa21f7e79d"
                           "32267420ca8d7028d97673de5342ae97de8057), verbatim "
                           "constants and per-(slice,condition) single-mask reuse",
        "checkpoints": {"VN": CKPT_VARNET_SHA256, "PMR": CKPT_PMR_SHA256,
                        "PMRP": CKPT_PMRP_SHA256},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__,
                        "skimage_vendored": "0.25.2 (original run: fastmri_deps, "
                                            "version unrecorded; affects SSIM only)",
                        "platform": platform.platform(),
                        "device": torch.cuda.get_device_name(0)
                        if device.type == "cuda" else "cpu"},
        "gates": gates,
        "worst": {"dPSNR": worst_psnr, "dNMSE_rel": worst_nmse, "dSSIM": worst_ssim},
        "tolerances": {"PSNR_TOL_DB": PSNR_TOL_DB, "NMSE_TOL_REL": NMSE_TOL_REL,
                       "note": "pre-declared suggestions for this regeneration, "
                               "not community standards"},
        "comparison_24_cells": comparison,
        "npz_keys": sorted(qual.keys()),
        "npz_path": NPZ_OUT,
        "display_intent": "slice idx 7 (1-based 8); six-panel C1 rho=0.25/1.00 "
                          "x VN/PMR/PMRP with unified ROI zoom + blue error maps",
        "runtime_seconds": time.perf_counter() - t0,
    }
    with open(JSON_OUT, "w") as f:
        json.dump(record, f, indent=1)
    print(f"[COMPLETE] worst dPSNR={worst_psnr:.4f} dB, worst dNMSE_rel="
          f"{worst_nmse:.4%}, worst dSSIM={worst_ssim:.2e} "
          f"({record['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
