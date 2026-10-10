"""Q2 SSRE 001 -- Sensitivity-Support Robustness Evaluation (SSRE) matrix on the
reused 68-volume fastMRI brain AXT2 validation subset: 3 frozen public
checkpoints x 2 acquisition conditions (R=4 / R=8) x 4 prespecified normalized
sensitivity-support points, with studentized simultaneous-band ranking verdicts
and the frozen Table 9 terminal classification.

Authorization state (2026-08-31):
  - Preregistration (frozen BEFORE any new science PSNR forward):
    D:/lw3/docs/Q2_SSRE_PREREGISTRATION_001.md
  - Experiment guide (execution authority):
    D:/lw3/docs/paper/Q2_SSRE_EXPERIMENT_GUIDE_001.md
  - Design lock: D:/lw3/docs/paper/Q2_SSRE_RECAST_SINGLE_CORE_DESIGN_001.md
    (OWNER_APPROVED: YES).
  - Sample: REUSE of the G3 confirmation 68-volume list (within-sample method
    extension; NOT a new analysis-independent cohort). PromptMR v1 is the
    second official weight of the PromptMR family (same family; must be
    disclosed), not a third architecture.
  - Sole intervention per (volume, condition): the num_low_frequencies
    argument fed to each model's official sensitivity stage. The acquisition
    mask, masked k-space (within each model's official spatial preprocessing),
    target, 320x320 crop and official metric implementation are bitwise
    identical across models and rho within a (volume, condition).
  - Pipeline inheritance: VN and PMRP forward / crop / official metrics are
    verbatim from the reviewed
    D:/lw3/complex_mri_audit/scripts/g3_analysis_independent_confirmation_001.py
    (which itself verbatim inherits g3_post_e2_minimal_extension_001.py).
    PMR v1 uses the SAME official repo's models/promptmr.py:PromptMR loaded
    through the checkpoint's own hyper_parameters (model_version="promptmr",
    num_adj_slices=3, adaptive_input=False, n_buffer=0) -- official 3-slice
    clamped stacking per configs/inference/pmr/fm-brain.yaml. No shim is
    needed or permitted for v1. PMRP keeps ONLY the recorded upstream-compat
    shim blk.n_buffer = int(blk.model.n_buffer) = 4.
  - fp32, torch.inference_mode(), no autocast/quantization; cudnn.benchmark
    fixed to False. Primary metric: fastmri.evaluate.psnr on (16,320,320)
    stacks (gt first, data_range=gt.max()), one value per volume. NMSE/SSIM
    are secondary (JSON/appendix only; never enter Table 9).
  - Statistics: B=10000, seed=20260901, resampling unit = volume (each draw
    resamples the volume's whole model x rho vector). Studentized max-t
    simultaneous 95% band = frozen CALIBRATION SUGGESTION for this experiment,
    not a community standard. Pointwise 1.96*SE is descriptive only and never
    decides dominance/crossing/Table 9.
  - Failure of any gate: print STOP, write the stop record to the NEW report
    path only, exit 1. No repair, no rerun, no second review. Smoke mode
    writes NO science JSON and never enters Table 9.

Modes:
  --mode smoke     : Phase-1 hard smoke on the three frozen 12/14/20-coil
                     volumes, one slice each: 3 models x {C1 rho=1 (L=32),
                     C2 rho=0.25 (L=4)}. Finite outputs, no OOM, mask purity,
                     num_low_frequencies-trace assertions. No JSON written.
  --mode preflight : identity bindings + frozen 68-volume identity
                     (metadata-only) + strict loads + static num_low_
                     frequencies traces + ACS-extractor pads + official mask
                     measurements (C1=123 cols; C2 nlf=16, per-seed sampled
                     cols) + mask-layer equality + synthetic forwards for
                     12/14/20 coils x all six L values x 3 models. Reads NO
                     candidate k-space/target arrays, writes NO report.
  --mode science   : all preflight gates, then the single formal 1632-cell
                     run -> frozen statistics -> mechanical Table 9 ->
                     D:/lw3/complex_mri_audit/reports/q2_ssre_001.json
                     (incrementally and atomically updated per volume; the
                     file must not exist before this run).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
import time

# --------------------------------------------------------------------------- #
# paths and frozen identities
# --------------------------------------------------------------------------- #
DEPS_DIR = r"D:/lw3/_external_baselines/fastmri_deps"          # vendored skimage
FASTMRI_REPO = r"D:/lw3/_external_baselines/fastMRI_official"  # commit 91f2df4...
PMRP_REPO = r"D:/lw3/_external_baselines/PromptMR_plus_official"
for _p in (DEPS_DIR, FASTMRI_REPO, PMRP_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import h5py  # noqa: E402

import fastmri  # noqa: E402
import fastmri.data.transforms as T  # noqa: E402
import fastmri.models.varnet as fastmri_varnet_module  # noqa: E402
from fastmri.data.subsample import create_mask_for_mask_type  # noqa: E402
from fastmri.data.transforms import VarNetDataTransform, apply_mask as fm_apply_mask  # noqa: E402
from fastmri.evaluate import nmse, psnr, ssim  # noqa: E402
from fastmri.models import VarNet  # noqa: E402

import data.transforms as PT  # noqa: E402  (official PromptMR repo 'data' package)
from data.transforms import FastmriDataTransform, apply_mask as pm_apply_mask  # noqa: E402
from models.utils import KspaceACSExtractor  # noqa: E402
from pl_modules import PromptMrModule  # noqa: E402
import models.promptmr as pmr_v1_module  # noqa: E402
import models.promptmr_v2 as pmr_v2_module  # noqa: E402

DATA_DIR = r"D:/lw3/fastmri_brain_val/multicoil_val"
G3_JSON_PATH = r"D:/lw3/complex_mri_audit/reports/g3_analysis_independent_confirmation_001.json"
CAL_JSON_PATH = r"D:/lw3/complex_mri_audit/reports/g3_workpackage_a_calibration_001.json"
E2_JSON_PATH = r"D:/lw3/complex_mri_audit/reports/g3_pre_e2_amended_confirmation_001.json"
PREREG_PATH = r"D:/lw3/docs/Q2_SSRE_PREREGISTRATION_001.md"
PREREG_SHA256 = "ca67e7c8e6b777f8ed9e6b2c9bd3aaf2b802bd0e8d30c625c9bac093f033e35d"
GUIDE_PATH = r"D:/lw3/docs/paper/Q2_SSRE_EXPERIMENT_GUIDE_001.md"
REPORT_PATH = r"D:/lw3/complex_mri_audit/reports/q2_ssre_001.json"
QUAL_NPZ_PATH = r"D:/lw3/complex_mri_audit/reports/q2_ssre_001_qualitative.npz"
INHERITED_SCRIPT = r"D:/lw3/complex_mri_audit/scripts/g3_analysis_independent_confirmation_001.py"
INHERITED_SCRIPT_SHA256 = "c981833d8e7ec572986936ad7ecc15d7d5e201cb3812d1bd1259181db2637dc8"
INHERITED_SCRIPT_2 = r"D:/lw3/complex_mri_audit/scripts/g3_post_e2_minimal_extension_001.py"
INHERITED_SCRIPT_2_SHA256 = "09345a41fbcac5371c1c384d336b2e8173d80df08aefc194e6a1fd31ea352bd7"

CKPT_VARNET_PATH = r"D:/lw3/_external_baselines/weights/brain_leaderboard_state_dict.pt"
CKPT_VARNET_SHA256 = "2cdfdde2fe662f8995316a7ba981d8d3f405d2a5fd82a61bfa1bc4e56062262f"
CKPT_PMR_PATH = r"D:/lw3/_external_baselines/weights/promptmr-epoch=44-step=2083275.ckpt"
CKPT_PMR_SHA256 = "67a30557acdab3c19f118ea3d0cd63cc51a32b92326f09fd37ca89156369954e"
CKPT_PMR_BYTES = 314103087
CKPT_PMRP_PATH = r"D:/Google download/promptmr-plus-epoch%3D44-step%3D1591830.ckpt"
CKPT_PMRP_SHA256 = "42722018604944c567c598ddf5c488d135793ef337359a1da03ad3d4301e177e"
PMRP_COMMIT = "934eeda6d4d18cd39e406fa1eee9e1f70603cb5e"
FASTMRI_COMMIT = "91f2df4711adbb6d643df1810f234e4abcf5881b"

SEED = 20260901
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260901
SE_FLOOR = 1e-12

MASK_TYPE = "equispaced"
USE_SEED = True
C1 = dict(center_fraction=0.08, acceleration=4, l_nom=32,
          grid=((0.25, 8), (0.50, 16), (0.75, 24), (1.00, 32)))
C2 = dict(center_fraction=0.04, acceleration=8, l_nom=16,
          grid=((0.25, 4), (0.50, 8), (0.75, 12), (1.00, 16)))
CONDITIONS = {"C1": C1, "C2": C2}
EXPECTED_MASK_COLS_C1 = 123
RHO_GRID = (0.25, 0.50, 0.75, 1.00)
PAIRS = (("PMRP", "VN"), ("PMR", "VN"), ("PMRP", "PMR"))
MODELS = ("VN", "PMR", "PMRP")

NUM_CASCADES, CHANS, POOLS = 12, 18, 4
SENS_CHANS, SENS_POOLS = 8, 4
MASK_CENTER = True
TARGET_KEY = "reconstruction_rss"
EVAL_CROP = (320, 320)
EXPECTED_SLICES = 16
EXPECTED_ROWS, EXPECTED_COLS = 768, 396
EXPECTED_ACQ = "AXT2"
N_CONFIRM = 68
STRATUM_EXPECTED = {12: 8, 14: 23, 20: 37}
PMRP_UNIFORM_RESOLUTION = (384, 384)   # official fm-brain inference config
PMRP_NUM_ADJ_SLICES = 5                # official (checkpoint hyper_parameters)
PMR_NUM_ADJ_SLICES = 3                 # official fm-brain inference yaml + checkpoint hp
SMOKE_VOLTS = (("file_brain_AXT2_201_2010540.h5", 12),
               ("file_brain_AXT2_209_2090023.h5", 14),
               ("file_brain_AXT2_200_2000257.h5", 20))
QUAL_VOLTS = ("file_brain_AXT2_209_2090204.h5",
              "file_brain_AXT2_209_6001085.h5",
              "file_brain_AXT2_210_6001922.h5")
QUAL_SLICE_IDX = 7  # 0-based, frozen
SCIENCE_WALL_BUDGET_S = int(7.7 * 3600)  # 8 h total envelope minus preflight margin

# frozen 68-volume list (G3 confirmation prereg v1.1 section 2; 12 -> 14 -> 20 coils)
CONFIRMATION_VOLUMES = (
    # 12-coil (8)
    ("file_brain_AXT2_201_2010540.h5", 12),
    ("file_brain_AXT2_201_2010559.h5", 12),
    ("file_brain_AXT2_209_2090058.h5", 12),
    ("file_brain_AXT2_209_2090330.h5", 12),
    ("file_brain_AXT2_209_6001045.h5", 12),
    ("file_brain_AXT2_209_6001085.h5", 12),
    ("file_brain_AXT2_209_6001237.h5", 12),
    ("file_brain_AXT2_209_6001433.h5", 12),
    # 14-coil (23)
    ("file_brain_AXT2_209_2090023.h5", 14),
    ("file_brain_AXT2_209_2090034.h5", 14),
    ("file_brain_AXT2_209_2090098.h5", 14),
    ("file_brain_AXT2_209_2090103.h5", 14),
    ("file_brain_AXT2_209_2090105.h5", 14),
    ("file_brain_AXT2_209_2090127.h5", 14),
    ("file_brain_AXT2_209_2090161.h5", 14),
    ("file_brain_AXT2_209_2090188.h5", 14),
    ("file_brain_AXT2_209_2090191.h5", 14),
    ("file_brain_AXT2_209_2090204.h5", 14),
    ("file_brain_AXT2_209_2090277.h5", 14),
    ("file_brain_AXT2_209_2090323.h5", 14),
    ("file_brain_AXT2_209_2090346.h5", 14),
    ("file_brain_AXT2_209_2090403.h5", 14),
    ("file_brain_AXT2_209_6000974.h5", 14),
    ("file_brain_AXT2_209_6000992.h5", 14),
    ("file_brain_AXT2_209_6001048.h5", 14),
    ("file_brain_AXT2_209_6001066.h5", 14),
    ("file_brain_AXT2_209_6001169.h5", 14),
    ("file_brain_AXT2_209_6001318.h5", 14),
    ("file_brain_AXT2_209_6001361.h5", 14),
    ("file_brain_AXT2_209_6001422.h5", 14),
    ("file_brain_AXT2_209_6001438.h5", 14),
    # 20-coil (37)
    ("file_brain_AXT2_200_2000257.h5", 20),
    ("file_brain_AXT2_200_2000270.h5", 20),
    ("file_brain_AXT2_200_2000327.h5", 20),
    ("file_brain_AXT2_200_2000635.h5", 20),
    ("file_brain_AXT2_200_6002090.h5", 20),
    ("file_brain_AXT2_200_6002098.h5", 20),
    ("file_brain_AXT2_200_6002173.h5", 20),
    ("file_brain_AXT2_200_6002181.h5", 20),
    ("file_brain_AXT2_200_6002193.h5", 20),
    ("file_brain_AXT2_200_6002244.h5", 20),
    ("file_brain_AXT2_200_6002478.h5", 20),
    ("file_brain_AXT2_200_6002484.h5", 20),
    ("file_brain_AXT2_200_6002657.h5", 20),
    ("file_brain_AXT2_201_2010159.h5", 20),
    ("file_brain_AXT2_201_2010185.h5", 20),
    ("file_brain_AXT2_201_2010220.h5", 20),
    ("file_brain_AXT2_201_2010327.h5", 20),
    ("file_brain_AXT2_201_2010372.h5", 20),
    ("file_brain_AXT2_201_2010393.h5", 20),
    ("file_brain_AXT2_201_2010504.h5", 20),
    ("file_brain_AXT2_201_2010527.h5", 20),
    ("file_brain_AXT2_201_2010533.h5", 20),
    ("file_brain_AXT2_201_2010552.h5", 20),
    ("file_brain_AXT2_204_2040061.h5", 20),
    ("file_brain_AXT2_210_2100218.h5", 20),
    ("file_brain_AXT2_210_2100228.h5", 20),
    ("file_brain_AXT2_210_2100286.h5", 20),
    ("file_brain_AXT2_210_2100318.h5", 20),
    ("file_brain_AXT2_210_2100327.h5", 20),
    ("file_brain_AXT2_210_6001542.h5", 20),
    ("file_brain_AXT2_210_6001599.h5", 20),
    ("file_brain_AXT2_210_6001610.h5", 20),
    ("file_brain_AXT2_210_6001768.h5", 20),
    ("file_brain_AXT2_210_6001805.h5", 20),
    ("file_brain_AXT2_210_6001813.h5", 20),
    ("file_brain_AXT2_210_6001822.h5", 20),
    ("file_brain_AXT2_210_6001922.h5", 20),
)
assert len(CONFIRMATION_VOLUMES) == N_CONFIRM
_stratum_counts = {}
for _f, _c in CONFIRMATION_VOLUMES:
    _stratum_counts[_c] = _stratum_counts.get(_c, 0) + 1
assert _stratum_counts == STRATUM_EXPECTED, _stratum_counts
CONFIRMATION_FILENAMES = frozenset(f for f, _ in CONFIRMATION_VOLUMES)
assert set(QUAL_VOLTS) <= CONFIRMATION_FILENAMES
assert {f for f, _ in SMOKE_VOLTS} <= CONFIRMATION_FILENAMES
for _cond, _cfg in CONDITIONS.items():
    assert tuple(r for r, _ in _cfg["grid"]) == RHO_GRID
    assert tuple(l for _, l in _cfg["grid"]) == tuple(int(round(r * _cfg["l_nom"])) for r, _ in _cfg["grid"])


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def official_metadata(fname):
    """Byte-for-byte equivalent of the locked SliceDataset._retrieve_metadata
    (same as the frozen E2/G3/inherited implementations; feeds ALL models)."""
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
    """Volume-level official metrics: fastmri.evaluate on the (16,320,320)
    stacks -- identical algorithm and statistical unit as the frozen E2/G3
    metric_set (gt first argument, data_range=gt.max())."""
    return {"NMSE": float(nmse(gt_stack, pred_stack).item()),
            "PSNR": float(psnr(gt_stack, pred_stack).item()),
            "SSIM": float(ssim(gt_stack, pred_stack).item())}


def mask_cols_offset(mask_bool_vec):
    return int(mask_bool_vec.sum().item()), int(mask_bool_vec.float().argmax().item())


# --------------------------------------------------------------------------- #
# model loading (strict)
# --------------------------------------------------------------------------- #
def load_varnet(device):
    model = VarNet(num_cascades=NUM_CASCADES, sens_chans=SENS_CHANS,
                   sens_pools=SENS_POOLS, chans=CHANS, pools=POOLS,
                   mask_center=MASK_CENTER).to(device)
    sd = torch.load(CKPT_VARNET_PATH, weights_only=True, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    ok = len(missing) == 0 and len(unexpected) == 0
    model.eval()
    assert all(p.dtype == torch.float32 for p in model.parameters()), "VarNet not fp32"
    return model, {"missing_keys": list(missing), "unexpected_keys": list(unexpected),
                   "strict_ok": bool(ok)}


def load_pmrp(device):
    ck = torch.load(CKPT_PMRP_PATH, weights_only=True, map_location="cpu")
    hp = dict(ck["hyper_parameters"])
    sd = {k: v for k, v in ck["state_dict"].items()}
    meta = {"epoch": int(ck["epoch"]), "global_step": int(ck["global_step"]),
            "pytorch_lightning_version": str(ck["pytorch-lightning_version"]),
            "state_dict_entries": len(sd)}
    module = PromptMrModule(**hp)
    missing, unexpected = module.load_state_dict(sd, strict=True)
    strict_ok = len(missing) == 0 and len(unexpected) == 0
    # inherited upstream-compat shim (the ONLY permitted shim; recorded since E2):
    # official HEAD PromptMRBlock.forward references self.n_buffer, which exists
    # only on self.model. Checkpoint-era official code composes the same buffer
    # for n_buffer=4.
    shim = {"blocks_patched": 0, "values": set()}
    for blk in module.promptmr.cascades:
        assert not hasattr(blk, "n_buffer"), "unexpected pre-existing blk.n_buffer"
        assert hasattr(blk.model, "n_buffer"), "NormPromptUnet.n_buffer missing"
        blk.n_buffer = int(blk.model.n_buffer)
        shim["blocks_patched"] += 1
        shim["values"].add(blk.n_buffer)
    shim["values"] = sorted(shim["values"])
    shim["ok"] = shim["blocks_patched"] == NUM_CASCADES and shim["values"] == [hp["n_buffer"]]
    module.eval()
    module.to(device)
    hp_public = {k: (list(v) if isinstance(v, tuple) else v) for k, v in hp.items()}
    return module, {"hyper_parameters": hp_public, "checkpoint_meta": meta,
                    "missing_keys": list(missing), "unexpected_keys": list(unexpected),
                    "strict_ok": bool(strict_ok), "n_buffer_shim": shim,
                    "model_class": "models/promptmr_v2.py:PromptMR"}


def load_pmr(device):
    """Official PromptMR v1: loaded through its OWN checkpoint hyper_parameters
    (model_version='promptmr' dynamically selects models/promptmr.py:PromptMR).
    num_adj_slices=3 comes from the checkpoint itself and matches the official
    configs/inference/pmr/fm-brain.yaml. adaptive_input=False, n_buffer=0,
    n_history=0: the official PromptMR.forward composes buffer=None internally;
    NO shim is needed and none is applied."""
    ck = torch.load(CKPT_PMR_PATH, weights_only=True, map_location="cpu")
    hp = dict(ck["hyper_parameters"])
    assert hp["model_version"] == "promptmr", hp["model_version"]
    assert int(hp["num_adj_slices"]) == PMR_NUM_ADJ_SLICES, hp["num_adj_slices"]
    assert bool(hp["adaptive_input"]) is False and int(hp["n_buffer"]) == 0
    sd = {k: v for k, v in ck["state_dict"].items()}
    meta = {"epoch": int(ck["epoch"]), "global_step": int(ck["global_step"]),
            "pytorch_lightning_version": str(ck["pytorch-lightning_version"]),
            "state_dict_entries": len(sd)}
    module = PromptMrModule(**hp)
    missing, unexpected = module.load_state_dict(sd, strict=True)
    strict_ok = len(missing) == 0 and len(unexpected) == 0
    no_shim = {"blocks_patched": 0, "values": [0], "ok": True,
               "reason": ("v1 PromptMR.forward composes buffer internally from "
                          "self.n_buffer=0; PromptMRBlock.forward never references "
                          "n_buffer; no adaptation needed or applied")}
    for blk in module.promptmr.cascades:
        assert not hasattr(blk, "n_buffer"), "v1 must carry no blk.n_buffer shim"
    assert int(module.promptmr.n_buffer) == 0
    module.eval()
    module.to(device)
    hp_public = {k: (list(v) if isinstance(v, tuple) else v) for k, v in hp.items()}
    return module, {"hyper_parameters": hp_public, "checkpoint_meta": meta,
                    "missing_keys": list(missing), "unexpected_keys": list(unexpected),
                    "strict_ok": bool(strict_ok), "n_buffer_shim": no_shim,
                    "model_class": "models/promptmr.py:PromptMR"}


# --------------------------------------------------------------------------- #
# static num_low_frequencies trace: intervention may enter ONLY the
# sensitivity stage (preregistration; guide section 6.3)
# --------------------------------------------------------------------------- #
def _nlf_uses_outside_signature(func):
    """AST-based: num_low_frequencies appears in a signature as ast.arg (not
    ast.Name), so every ast.Name hit below is a body reference; docstring text
    is a Constant and is also not a Name. Returns (lineno, col) pairs."""
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    uses = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "num_low_frequencies":
            uses.append((node.lineno, node.col_offset))
    return uses


def _assert_forward_nlf_in_sens_stage_only(func, name):
    uses = _nlf_uses_outside_signature(func)
    src = textwrap.dedent(inspect.getsource(func))
    lines = src.splitlines()
    sens_net_lines = {i for i, line in enumerate(lines, start=1) if "sens_net" in line}
    assert uses, f"{name}: sensitivity-stage num_low_frequencies usage not found"
    for lineno, _ in uses:
        assert lineno in sens_net_lines, (
            f"{name}: num_low_frequencies used at line {lineno} outside the "
            f"sens_net call (sens_net lines: {sorted(sens_net_lines)})")
    return {"function": name, "nlf_use_lines": [ln for ln, _ in uses],
            "sens_net_lines": sorted(sens_net_lines), "pass": True}


def _assert_forward_nlf_absent(func, name):
    uses = _nlf_uses_outside_signature(func)
    assert not uses, f"{name}: unexpected num_low_frequencies usage at {[u[0] for u in uses]}"
    return {"function": name, "nlf_use_lines": [], "pass": True}


def static_trace_checks():
    checks = {}
    checks["varnet_forward"] = _assert_forward_nlf_in_sens_stage_only(
        fastmri_varnet_module.VarNet.forward, "fastmri.models.varnet.VarNet.forward")
    checks["varnet_block"] = _assert_forward_nlf_absent(
        fastmri_varnet_module.VarNetBlock.forward, "fastmri.models.varnet.VarNetBlock.forward")
    checks["pmr_v1_forward"] = _assert_forward_nlf_in_sens_stage_only(
        pmr_v1_module.PromptMR.forward, "models/promptmr.py:PromptMR.forward")
    checks["pmr_v1_block"] = _assert_forward_nlf_absent(
        pmr_v1_module.PromptMRBlock.forward, "models/promptmr.py:PromptMRBlock.forward")
    checks["pmrp_v2_forward"] = _assert_forward_nlf_in_sens_stage_only(
        pmr_v2_module.PromptMR.forward, "models/promptmr_v2.py:PromptMR.forward")
    checks["pmrp_v2_block"] = _assert_forward_nlf_absent(
        pmr_v2_module.PromptMRBlock.forward, "models/promptmr_v2.py:PromptMRBlock.forward")
    checks["all_pass"] = bool(all(v["pass"] for v in checks.values()))
    return checks


def acs_extractor_pad_check():
    """Intervention entry point on the PromptMR side: the ACS extractor pads the
    masked k-space down to exactly L center columns before the sensitivity
    stage (same arithmetic as the reviewed G3 check, extended to all six L)."""
    ex = KspaceACSExtractor(True)
    out = {}
    w = EXPECTED_COLS
    ok_all = True
    for L in (4, 8, 12, 16, 24, 32):
        dummy_mask = torch.zeros(1, 1, 1, w, 1, dtype=torch.bool)
        dummy_mask[..., w // 2 - L // 2: w // 2 - L // 2 + L, :] = True
        pad, nl = ex.get_pad_and_num_low_freqs(dummy_mask, torch.tensor([L]))
        ok = int(nl) == L
        out[f"L{L}"] = {"pad": int(pad), "num_low_frequencies": int(nl),
                        "expected_pad": ACS_PAD_EXPECTED[L], "pass": ok}
        ok_all = ok_all and ok
    out["all_pass"] = bool(ok_all)
    return out


ACS_PAD_EXPECTED = {L: (EXPECTED_COLS - L + 1) // 2 for L in (4, 8, 12, 16, 24, 32)}


# --------------------------------------------------------------------------- #
# frozen-sample identity verification (metadata-only)
# --------------------------------------------------------------------------- #
def read_used30_names():
    cal = json.load(open(CAL_JSON_PATH, encoding="utf-8"))
    e2 = json.load(open(E2_JSON_PATH, encoding="utf-8"))
    dev6 = set(cal["inventory"]["development_volumes"]) | \
        set(cal["access_control"]["forward_files_used"])
    conf24 = set(e2["access_control"]["confirmation_kspace_read_files"])
    return sorted(dev6 | conf24)


def verify_frozen_sample(mask_func_c1):
    """Metadata-only verification of the reused 68-volume identity, coil
    strata, and the C1 observed mask offsets (cross-checked against the frozen
    G3 confirmation JSON). Reads only shapes + attrs; no k-space/target read."""
    g3 = json.load(open(G3_JSON_PATH, encoding="utf-8"))
    g3_meta = g3["e2_volume_metadata"]
    per_file = {}
    strata = {12: 0, 14: 0, 20: 0}
    all_ok = True
    for fname, exp_coils in CONFIRMATION_VOLUMES:
        path = os.path.join(DATA_DIR, fname)
        with h5py.File(path, "r") as hf:  # metadata-only open
            ks = tuple(int(v) for v in hf["kspace"].shape)
            ts = tuple(int(v) for v in hf[TARGET_KEY].shape)
            attrs = {k: (v.decode() if isinstance(v, bytes) else v)
                     for k, v in hf.attrs.items()}
        acq = str(attrs.get("acquisition"))
        pid = str(attrs.get("patient_id", ""))
        m, _ = mask_func_c1((1, 1, EXPECTED_COLS, 2), None, tuple(map(ord, fname)))
        off = int(m.numpy().reshape(-1).argmax())
        g3_entry = g3_meta.get(fname)
        consistent_with_g3 = bool(
            g3_entry is not None
            and g3_entry["acquisition"] == acq
            and g3_entry["patient_id"] == pid
            and g3_entry["coils"] == exp_coils
            and g3_entry["observed_mask_offset"] == off
            and bool(g3_entry["structure_ok"]))
        ok = (acq == EXPECTED_ACQ
              and ks == (EXPECTED_SLICES, exp_coils, EXPECTED_ROWS, EXPECTED_COLS)
              and ts == (EXPECTED_SLICES, 384, 384)
              and consistent_with_g3)
        all_ok = all_ok and ok
        strata[exp_coils] += 1
        per_file[fname] = {"acquisition": acq, "patient_id": pid, "coils": exp_coils,
                           "kspace_shape": list(ks), "target_shape": list(ts),
                           "observed_mask_offset_C1": off,
                           "consistent_with_g3_confirmation": consistent_with_g3,
                           "structure_ok": ok}
    pids68 = [v["patient_id"] for v in per_file.values()]
    pids_unique = len(set(pids68)) == N_CONFIRM
    return {"per_file": per_file, "strata_observed": strata,
            "strata_ok": strata == STRATUM_EXPECTED, "all_structure_ok": bool(all_ok),
            "n": len(per_file), "n_ok": len(per_file) == N_CONFIRM,
            "patient_id_unique_within_68": bool(pids_unique),
            "all_metadata_ok": bool(all_ok and pids_unique
                                    and strata == STRATUM_EXPECTED
                                    and len(per_file) == N_CONFIRM)}


# --------------------------------------------------------------------------- #
# official mask measurement (mask level only; no data arrays)
# --------------------------------------------------------------------------- #
def mask_func_for(cond_name):
    cfg = CONDITIONS[cond_name]
    return create_mask_for_mask_type(MASK_TYPE, [cfg["center_fraction"]],
                                     [cfg["acceleration"]])


def measure_official_masks():
    """C1 must give 123 sampled columns for every filename; C2's sampled column
    count is whatever the OFFICIAL MaskFunc produces per filename seed (measured,
    never guessed; preregistration appendix A) and num_low_frequencies must be
    16 for every filename."""
    out = {"per_file": {}, "c1_all_123": True, "c2_all_nlf16": True,
           "c2_cols_observed": {}, "pass": True}
    for cond_name in ("C1", "C2"):
        mf = mask_func_for(cond_name)
        cfg = CONDITIONS[cond_name]
        for fname, _ in CONFIRMATION_VOLUMES:
            m, nlf = mf((1, 1, EXPECTED_COLS, 2), None, tuple(map(ord, fname)))
            mv = m.numpy().reshape(-1)
            cols, off = int(mv.sum()), int(mv.argmax())
            out["per_file"].setdefault(fname, {})[cond_name] = {
                "sampled_cols": cols, "offset": off, "num_low_frequencies": int(nlf)}
            if cond_name == "C1":
                out["c1_all_123"] &= (cols == EXPECTED_MASK_COLS_C1 and int(nlf) == cfg["l_nom"])
            else:
                out["c2_all_nlf16"] &= (int(nlf) == cfg["l_nom"])
                out["c2_cols_observed"][cols] = out["c2_cols_observed"].get(cols, 0) + 1
    out["pass"] = bool(out["c1_all_123"] and out["c2_all_nlf16"])
    return out


# --------------------------------------------------------------------------- #
# mask-layer equality across the two official apply_mask implementations
# --------------------------------------------------------------------------- #
def mask_layer_equality_check():
    out = {}
    ok_all = True
    for cond_name in ("C1", "C2"):
        mf = mask_func_for(cond_name)
        cfg = CONDITIONS[cond_name]
        seed = tuple(map(ord, CONFIRMATION_VOLUMES[0][0]))
        rng = np.random.default_rng(SEED)
        ks_syn = (rng.standard_normal((12, EXPECTED_ROWS, EXPECTED_COLS))
                  + 1j * rng.standard_normal((12, EXPECTED_ROWS, EXPECTED_COLS))).astype(np.complex64)
        ks_t = T.to_tensor(ks_syn)
        fm_mks, fm_mask, fm_nlf = fm_apply_mask(ks_t.clone(), mf, seed=seed,
                                                padding=(0, EXPECTED_COLS))
        pm_mks, pm_mask, pm_nlf, pm_mt = pm_apply_mask(ks_t.clone(), mf, seed=seed,
                                                       padding=(0, EXPECTED_COLS))
        ok = (torch.equal(fm_mask, pm_mask) and torch.equal(fm_mks, pm_mks)
              and int(fm_nlf) == cfg["l_nom"] and int(pm_nlf) == cfg["l_nom"])
        out[cond_name] = {"mask_equal": bool(torch.equal(fm_mask, pm_mask)),
                          "masked_kspace_equal": bool(torch.equal(fm_mks, pm_mks)),
                          "num_low_frequencies": [int(fm_nlf), int(pm_nlf)],
                          "pmrp_mask_type": str(pm_mt), "pass": bool(ok)}
        ok_all = ok_all and ok
    out["all_pass"] = bool(ok_all)
    return out


# --------------------------------------------------------------------------- #
# synthetic forwards: every stratum coil count x every frozen L x 3 models
# (pure synthetic tensors; no candidate volume read)
# --------------------------------------------------------------------------- #
def synthetic_forwards_check(varnet, pmr, pmrp, device):
    checks = {"per_coils": {}, "peak_gpu_mem_gb": 0.0}
    ok_all = True
    seed = tuple(map(ord, CONFIRMATION_VOLUMES[0][0]))
    rng = np.random.default_rng(SEED)
    grid_L = sorted({l for _, l in C1["grid"]} | {l for _, l in C2["grid"]})
    for coils in sorted(STRATUM_EXPECTED):
        # VarNet: full 768x396, official path
        ks_v = (rng.standard_normal((coils, EXPECTED_ROWS, EXPECTED_COLS))
                + 1j * rng.standard_normal((coils, EXPECTED_ROWS, EXPECTED_COLS))).astype(np.complex64)
        mks, msk, _ = fm_apply_mask(T.to_tensor(ks_v), mask_func_for("C1"), seed=seed,
                                    padding=(0, EXPECTED_COLS))
        mks_b = mks.unsqueeze(0).to(device)
        msk_b = msk.to(torch.bool).unsqueeze(0).to(device)
        mks_snapshot = mks_b.clone()
        vn_ok = True
        crops = {}
        with torch.inference_mode():
            assert not torch.is_autocast_enabled(), "autocast must stay disabled"
            for L in grid_L:
                kpred = varnet_fwd(varnet, mks_b, msk_b, L)
                mag = magnitude_from_kspace(kpred)
                crop = T.center_crop(mag, EVAL_CROP)
                vn_ok = vn_ok and (list(mag.shape) == [1, EXPECTED_ROWS, EXPECTED_COLS]
                                   and list(crop.shape) == [1, EVAL_CROP[0], EVAL_CROP[1]]
                                   and bool(torch.isfinite(crop).all()))
                crops[L] = crop
            vn_ok = vn_ok and (not torch.equal(crops[min(grid_L)], crops[max(grid_L)]))
            vn_ok = vn_ok and bool(torch.equal(mks_b, mks_snapshot))  # inputs never mutated
        del kpred, mag, crop, crops
        # PromptMR v1: official 3-slice stacked channels, height-cropped 384 rows
        ks_p3 = (rng.standard_normal((coils * PMR_NUM_ADJ_SLICES, 384, EXPECTED_COLS))
                 + 1j * rng.standard_normal((coils * PMR_NUM_ADJ_SLICES, 384, EXPECTED_COLS))).astype(np.complex64)
        mks3, msk3, nlf3 = fm_apply_mask(T.to_tensor(ks_p3), mask_func_for("C1"), seed=seed,
                                         padding=(0, EXPECTED_COLS))
        ks_b3 = mks3.unsqueeze(0).to(device)
        mk_b3 = msk3.to(torch.bool).unsqueeze(0).to(device)
        ks3_snapshot = ks_b3.clone()
        pmr_ok = True
        outs3 = {}
        with torch.inference_mode():
            for L in grid_L:
                od = pmr(ks_b3, mk_b3, torch.tensor([L], device=device), ["cartesian"])
                img = od["img_pred"]
                crop = T.center_crop(PT.center_crop(img, PMRP_UNIFORM_RESOLUTION), EVAL_CROP)
                pmr_ok = pmr_ok and (list(img.shape) == [1, 384, EXPECTED_COLS]
                                     and list(crop.shape) == [1, EVAL_CROP[0], EVAL_CROP[1]]
                                     and bool(torch.isfinite(img).all()))
                outs3[L] = img
            pmr_ok = pmr_ok and (not torch.equal(outs3[min(grid_L)], outs3[max(grid_L)]))
            pmr_ok = pmr_ok and bool(torch.equal(ks_b3, ks3_snapshot))
        del od, img, crop, outs3
        # PromptMR+: official 5-slice stacked channels
        ks_p5 = (rng.standard_normal((coils * PMRP_NUM_ADJ_SLICES, 384, EXPECTED_COLS))
                 + 1j * rng.standard_normal((coils * PMRP_NUM_ADJ_SLICES, 384, EXPECTED_COLS))).astype(np.complex64)
        mks5, msk5, nlf5 = fm_apply_mask(T.to_tensor(ks_p5), mask_func_for("C1"), seed=seed,
                                         padding=(0, EXPECTED_COLS))
        ks_b5 = mks5.unsqueeze(0).to(device)
        mk_b5 = msk5.to(torch.bool).unsqueeze(0).to(device)
        pmrp_ok = True
        outs5 = {}
        with torch.inference_mode():
            for L in grid_L:
                od = pmrp(ks_b5, mk_b5, torch.tensor([L], device=device), ["cartesian"])
                img = od["img_pred"]
                crop = T.center_crop(PT.center_crop(img, PMRP_UNIFORM_RESOLUTION), EVAL_CROP)
                pmrp_ok = pmrp_ok and (list(img.shape) == [1, 384, EXPECTED_COLS]
                                       and list(crop.shape) == [1, EVAL_CROP[0], EVAL_CROP[1]]
                                       and bool(torch.isfinite(img).all()))
                outs5[L] = img
            pmrp_ok = pmrp_ok and (not torch.equal(outs5[min(grid_L)], outs5[max(grid_L)]))
        del od, img, crop, outs5
        assert int(nlf3) == 32 and int(nlf5) == 32
        if device.type == "cuda":
            checks["peak_gpu_mem_gb"] = max(
                checks["peak_gpu_mem_gb"], torch.cuda.max_memory_allocated() / 2 ** 30)
            torch.cuda.empty_cache()
        checks["per_coils"][f"coils_{coils}"] = {
            "varnet_ok": bool(vn_ok), "pmr_v1_ok": bool(pmr_ok), "pmrp_ok": bool(pmrp_ok),
            "L_values": list(grid_L)}
        ok_all = ok_all and vn_ok and pmr_ok and pmrp_ok
    checks["all_pass"] = bool(ok_all)
    checks["algorithm"] = ("pure synthetic tensors (np.random.default_rng(20260901)); "
                           "every frozen L in {4,8,12,16,24,32} x {VN, PMR v1, PMRP} x "
                           "12/14/20 coils; outputs must differ across L extremes "
                           "(intervention visible); inputs must never be mutated")
    return checks


# --------------------------------------------------------------------------- #
# shared preflight gates
# --------------------------------------------------------------------------- #
def run_identity_bindings():
    identity = {}
    identity["varnet_ckpt_sha256"] = sha256_of(CKPT_VARNET_PATH)
    identity["varnet_ckpt_ok"] = identity["varnet_ckpt_sha256"] == CKPT_VARNET_SHA256
    identity["pmr_ckpt_sha256"] = sha256_of(CKPT_PMR_PATH)
    identity["pmr_ckpt_ok"] = identity["pmr_ckpt_sha256"] == CKPT_PMR_SHA256
    identity["pmr_ckpt_bytes"] = os.path.getsize(CKPT_PMR_PATH)
    identity["pmr_ckpt_bytes_ok"] = identity["pmr_ckpt_bytes"] == CKPT_PMR_BYTES
    identity["pmrp_ckpt_sha256"] = sha256_of(CKPT_PMRP_PATH)
    identity["pmrp_ckpt_ok"] = identity["pmrp_ckpt_sha256"] == CKPT_PMRP_SHA256
    identity["pmrp_commit"] = subprocess.run(
        ["git", "-C", PMRP_REPO, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    identity["pmrp_commit_ok"] = identity["pmrp_commit"] == PMRP_COMMIT
    identity["fastmri_commit"] = subprocess.run(
        ["git", "-C", FASTMRI_REPO, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    identity["fastmri_commit_ok"] = identity["fastmri_commit"] == FASTMRI_COMMIT
    identity["prereg_sha256"] = sha256_of(PREREG_PATH)
    identity["prereg_ok"] = identity["prereg_sha256"] == PREREG_SHA256
    identity["inherited_script_sha256"] = sha256_of(INHERITED_SCRIPT)
    identity["inherited_script_ok"] = identity["inherited_script_sha256"] == INHERITED_SCRIPT_SHA256
    identity["inherited_script2_sha256"] = sha256_of(INHERITED_SCRIPT_2)
    identity["inherited_script2_ok"] = identity["inherited_script2_sha256"] == INHERITED_SCRIPT_2_SHA256
    identity_ok = (identity["varnet_ckpt_ok"] and identity["pmr_ckpt_ok"]
                   and identity["pmr_ckpt_bytes_ok"] and identity["pmrp_ckpt_ok"]
                   and identity["pmrp_commit_ok"] and identity["fastmri_commit_ok"]
                   and identity["prereg_ok"] and identity["inherited_script_ok"]
                   and identity["inherited_script2_ok"])
    return identity, identity_ok


def run_all_gates(need_models=True):
    """Identity + frozen sample + strict loads + static + mask + synthetic.
    Returns (gates, models_or_None, payload)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    identity, identity_ok = run_identity_bindings()
    print(f"[identity] varnet_ok={identity['varnet_ckpt_ok']} "
          f"pmr_ok={identity['pmr_ckpt_ok']} pmrp_ok={identity['pmrp_ckpt_ok']} "
          f"commit_ok={identity['pmrp_commit_ok'] and identity['fastmri_commit_ok']} "
          f"prereg_ok={identity['prereg_ok']} inherited_ok="
          f"{identity['inherited_script_ok'] and identity['inherited_script2_ok']}", flush=True)
    if not identity_ok:
        print("[STOP] identity binding failed. No fix, no rerun.")
        return {"identity_bindings": False}, None, None

    frozen = verify_frozen_sample(mask_func_for("C1"))
    print(f"[frozen sample] n={frozen['n']} strata={frozen['strata_observed']} "
          f"structure_ok={frozen['all_structure_ok']} "
          f"pid_unique={frozen['patient_id_unique_within_68']}", flush=True)
    if not frozen["all_metadata_ok"]:
        print("[STOP] reused 68-volume identity/strata failed. No fix, no rerun.")
        return {"frozen_sample_metadata": False}, None, None

    loads = {}
    if need_models:
        varnet, loads["varnet"] = load_varnet(device)
        print(f"[varnet] strict_ok={loads['varnet']['strict_ok']}", flush=True)
        if not loads["varnet"]["strict_ok"]:
            print("[STOP] VarNet strict load failed. No fix, no rerun.")
            return {"strict_loads": False}, None, None
        pmr, loads["pmr"] = load_pmr(device)
        print(f"[pmr] strict_ok={loads['pmr']['strict_ok']} "
              f"adj_slices={loads['pmr']['hyper_parameters']['num_adj_slices']} "
              f"n_buffer={loads['pmr']['hyper_parameters']['n_buffer']}", flush=True)
        if not loads["pmr"]["strict_ok"]:
            print("[STOP] PromptMR v1 strict load failed. No fix, no rerun.")
            return {"strict_loads": False}, None, None
        pmrp, loads["pmrp"] = load_pmrp(device)
        print(f"[pmrp] strict_ok={loads['pmrp']['strict_ok']} "
              f"n_buffer_shim={loads['pmrp']['n_buffer_shim']}", flush=True)
        if not (loads["pmrp"]["strict_ok"] and loads["pmrp"]["n_buffer_shim"]["ok"]):
            print("[STOP] PromptMR+ strict load or shim consistency failed. No fix, no rerun.")
            return {"strict_loads": False, "n_buffer_shim_consistent": False}, None, None

    traces = static_trace_checks()
    print(f"[static trace] nlf-in-sens-stage-only all_pass={traces['all_pass']}", flush=True)
    pads = acs_extractor_pad_check()
    print(f"[acs extractor] all_pass={pads['all_pass']}", flush=True)
    masks = measure_official_masks()
    print(f"[official masks] c1_all_123={masks['c1_all_123']} "
          f"c2_all_nlf16={masks['c2_all_nlf16']} "
          f"c2_cols_observed={masks['c2_cols_observed']}", flush=True)
    layer = mask_layer_equality_check()
    print(f"[mask layer equality] all_pass={layer['all_pass']}", flush=True)
    if not (traces["all_pass"] and pads["all_pass"] and masks["pass"] and layer["all_pass"]):
        print("[STOP] static/mask checks failed. No fix, no rerun.")
        return {"static_checks": False}, None, None

    synth = None
    if need_models:
        synth = synthetic_forwards_check(varnet, pmr, pmrp, device)
        print(f"[synthetic forwards] all_pass={synth['all_pass']} "
              f"peak_gpu={synth['peak_gpu_mem_gb']:.3f} GB", flush=True)
        if not synth["all_pass"]:
            print("[STOP] synthetic forwards failed. No fix, no rerun.")
            return {"synthetic_forwards": False}, None, None

    gates = {
        "identity_bindings": bool(identity_ok),
        "frozen_sample_metadata": bool(frozen["all_metadata_ok"]),
        "strict_loads": bool(loads["varnet"]["strict_ok"] and loads["pmr"]["strict_ok"]
                             and loads["pmrp"]["strict_ok"]),
        "n_buffer_shim_consistent": bool(loads["pmrp"]["n_buffer_shim"]["ok"]
                                         and loads["pmr"]["n_buffer_shim"]["ok"]),
        "static_trace_sens_stage_only": bool(traces["all_pass"]),
        "acs_extractor_pads": bool(pads["all_pass"]),
        "official_mask_measurements": bool(masks["pass"]),
        "mask_layer_equality": bool(layer["all_pass"]),
        "synthetic_forwards": bool(synth["all_pass"]) if synth is not None else None,
    }
    models = {"VN": varnet, "PMR": pmr, "PMRP": pmrp, "loads": loads, "device": device}
    payload = {"identity": identity, "frozen": frozen, "masks": masks, "traces": traces,
               "pads": pads, "layer": layer, "synth": synth}
    return gates, models, payload


# --------------------------------------------------------------------------- #
# smoke mode (Phase 1; real data, one slice per volume; NO science JSON)
# --------------------------------------------------------------------------- #
def run_smoke(models):
    varnet, pmr, pmrp = models["VN"], models["PMR"], models["PMRP"]
    device = models["device"]
    t_vn = {"C1": VarNetDataTransform(mask_func=mask_func_for("C1"), use_seed=USE_SEED),
            "C2": VarNetDataTransform(mask_func=mask_func_for("C2"), use_seed=USE_SEED)}
    t_pm = {"C1": FastmriDataTransform(mask_func=mask_func_for("C1"),
                                       uniform_resolution=PMRP_UNIFORM_RESOLUTION,
                                       use_seed=USE_SEED, mask_type=None,
                                       test_num_low_frequencies=None),
            "C2": FastmriDataTransform(mask_func=mask_func_for("C2"),
                                       uniform_resolution=PMRP_UNIFORM_RESOLUTION,
                                       use_seed=USE_SEED, mask_type=None,
                                       test_num_low_frequencies=None)}
    smoke_L = {"C1": 32, "C2": 4}  # C1 rho=1.00 ; C2 rho=0.25 (hard ACS=4 case)
    ok_all = True
    for fname, coils in SMOKE_VOLTS:
        path = os.path.join(DATA_DIR, fname)
        meta = official_metadata(path)
        with h5py.File(path, "r") as hf:
            kspace = hf["kspace"][0]
            target = hf[TARGET_KEY][0]
            for cond_name, L in smoke_L.items():
                cfg = CONDITIONS[cond_name]
                vs = t_vn[cond_name](kspace, None, target, meta, fname, 0)
                cols, off = mask_cols_offset(vs.mask[0, 0, :, 0])
                assert int(vs.num_low_frequencies) == cfg["l_nom"], \
                    f"{fname} {cond_name}: mask-derived nlf {vs.num_low_frequencies}"
                if cond_name == "C1":
                    assert cols == EXPECTED_MASK_COLS_C1, f"{fname} C1 cols {cols}"
                idx5 = [min(max(0 + j, 0), EXPECTED_SLICES - 1) for j in range(-2, 3)]
                stacked5 = np.concatenate([hf["kspace"][i] for i in idx5], axis=0)
                ps5 = t_pm[cond_name](stacked5, None, None, meta, fname, 0)
                idx3 = [min(max(0 + j, 0), EXPECTED_SLICES - 1) for j in range(-1, 2)]
                stacked3 = np.concatenate([hf["kspace"][i] for i in idx3], axis=0)
                ps3 = t_pm[cond_name](stacked3, None, None, meta, fname, 0)
                assert int(ps5.num_low_frequencies) == cfg["l_nom"]
                assert int(ps3.num_low_frequencies) == cfg["l_nom"]
                vn_mask_vec = vs.mask[0, 0, :, 0].cpu()
                mask_equal = (torch.equal(vn_mask_vec, ps5.mask[0, 0, :, 0].cpu())
                              and torch.equal(vn_mask_vec, ps3.mask[0, 0, :, 0].cpu()))
                assert mask_equal, f"{fname} {cond_name}: cross-model mask mismatch"
                vn_mks = vs.masked_kspace.unsqueeze(0).to(device)
                vn_mk = vs.mask.unsqueeze(0).to(device)
                p5_mks = ps5.masked_kspace.unsqueeze(0).to(device)
                p5_mk = ps5.mask.unsqueeze(0).to(device)
                p3_mks = ps3.masked_kspace.unsqueeze(0).to(device)
                p3_mk = ps3.mask.unsqueeze(0).to(device)
                snap = {"vn": vn_mks.clone(), "p5": p5_mks.clone(), "p3": p3_mks.clone()}
                with torch.inference_mode():
                    assert not torch.is_autocast_enabled()
                    kp = varnet_fwd(varnet, vn_mks, vn_mk, L)
                    vn_mag = T.center_crop(magnitude_from_kspace(kp), EVAL_CROP)
                    del kp
                    o5 = pmrp(p5_mks, p5_mk, torch.tensor([L], device=device), [ps5.mask_type])
                    p5_img = T.center_crop(PT.center_crop(o5["img_pred"], ps5.crop_size), EVAL_CROP)
                    del o5
                    o3 = pmr(p3_mks, p3_mk, torch.tensor([L], device=device), [ps3.mask_type])
                    p3_img = T.center_crop(PT.center_crop(o3["img_pred"], ps3.crop_size), EVAL_CROP)
                    del o3
                shape_ok = (list(vn_mag.shape) == [1] + list(EVAL_CROP)
                            and list(p5_img.shape) == [1] + list(EVAL_CROP)
                            and list(p3_img.shape) == [1] + list(EVAL_CROP))
                finite = (bool(torch.isfinite(vn_mag).all()) and bool(torch.isfinite(p5_img).all())
                          and bool(torch.isfinite(p3_img).all()))
                nomut = (torch.equal(vn_mks, snap["vn"]) and torch.equal(p5_mks, snap["p5"])
                         and torch.equal(p3_mks, snap["p3"]))
                mem = (torch.cuda.max_memory_allocated() / 2 ** 30) if device.type == "cuda" else 0.0
                cell_ok = bool(shape_ok and finite and nomut and mask_equal)
                ok_all = ok_all and cell_ok
                print(f"[smoke] {fname} c={coils} {cond_name} L={L}: shape={shape_ok} "
                      f"finite={finite} nomut={nomut} mask_eq={mask_equal} cols={cols} "
                      f"off={off} peak={mem:.2f} GB", flush=True)
                del vn_mks, vn_mk, p5_mks, p5_mk, p3_mks, p3_mk, vs, ps5, ps3
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    print(f"[smoke] ALL PASS = {ok_all}", flush=True)
    return ok_all


# --------------------------------------------------------------------------- #
# frozen statistics
# --------------------------------------------------------------------------- #
def simul_band(per_volume_matrix):
    """Studentized max-t simultaneous 95% band (frozen calibration suggestion,
    seed=20260901, B=10000, resampling unit = volume; a fresh
    np.random.default_rng(20260901) instance per curve so every curve is
    independently recomputable from the JSON's per-volume values)."""
    D = np.asarray(per_volume_matrix, dtype=np.float64)
    n, k = D.shape
    dhat = D.mean(axis=0)
    se = D.std(axis=0, ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_B, n))
    boot = D[idx]
    mean_b = boot.mean(axis=1)
    se_b = boot.std(axis=1, ddof=1) / np.sqrt(n)
    t_b = (np.abs(mean_b - dhat[None, :]) / np.maximum(se_b, SE_FLOOR)).max(axis=1)
    q = float(np.percentile(t_b, 95))
    lo = dhat - q * se
    hi = dhat + q * se
    return {"dhat": dhat, "se": se, "q": q, "lo": lo, "hi": hi,
            "pointwise_lo": dhat - 1.96 * se, "pointwise_hi": dhat + 1.96 * se}


def classify_curve(dhat, lo, hi):
    if bool(np.all(lo > 0)) or bool(np.all(hi < 0)):
        return "DOMINANCE"
    if bool(np.any(lo > 0)) and bool(np.any(hi < 0)):
        return "STRICT_CROSSING"
    if bool(np.any(dhat > 0)) and bool(np.any(dhat < 0)):
        return "DIRECTION_CHANGE_ONLY"
    return "UNRESOLVED"


# --------------------------------------------------------------------------- #
# science mode
# --------------------------------------------------------------------------- #
def run_science(models, gates, payload, t_start):
    varnet, pmr, pmrp = models["VN"], models["PMR"], models["PMRP"]
    device = models["device"]
    identity = payload["identity"]
    frozen = payload["frozen"]
    masks = payload["masks"]
    traces = payload["traces"]
    pads = payload["pads"]
    layer = payload["layer"]
    synth = payload["synth"]
    g3 = json.load(open(G3_JSON_PATH, encoding="utf-8"))
    g3_rows = {r["volume"]: r for r in g3["e2_per_volume"]}

    transforms_vn = {c: VarNetDataTransform(mask_func=mask_func_for(c), use_seed=USE_SEED)
                     for c in ("C1", "C2")}
    transforms_pm = {c: FastmriDataTransform(mask_func=mask_func_for(c),
                                             uniform_resolution=PMRP_UNIFORM_RESOLUTION,
                                             use_seed=USE_SEED, mask_type=None,
                                             test_num_low_frequencies=None)
                     for c in ("C1", "C2")}

    # psnr/nmse/ssim stores: [model][condition][rho] -> 68 volume values (frozen order)
    psnr_store = {m: {c: {r: [] for r in RHO_GRID} for c in ("C1", "C2")} for m in MODELS}
    nmse_store = {m: {c: {r: [] for r in RHO_GRID} for c in ("C1", "C2")} for m in MODELS}
    ssim_store = {m: {c: {r: [] for r in RHO_GRID} for c in ("C1", "C2")} for m in MODELS}
    rows = []
    qual = {}

    report = {
        "script": "run_q2_ssre_001",
        "mode": "science",
        "authorization": ("project owner design lock OWNER_APPROVED: YES "
                          "(Q2_SSRE_RECAST_SINGLE_CORE_DESIGN_001.md); execution per "
                          "Q2_SSRE_EXPERIMENT_GUIDE_001.md and "
                          "Q2_SSRE_PREREGISTRATION_001.md"),
        "run_nature": ("SSRE within-sample method extension on the REUSED 68-volume "
                       "fastMRI brain AXT2 validation subset; validation-subset only; "
                       "never test/leaderboard/clinical; PromptMR v1 is the second "
                       "official weight of the PromptMR family, not a third architecture"),
        "preregistration": {"path": PREREG_PATH, "sha256": PREREG_SHA256,
                            "frozen_before_any_new_science_forward": True},
        "guide": GUIDE_PATH,
        "frozen_bindings": {
            "varnet_ckpt_path": CKPT_VARNET_PATH,
            "varnet_ckpt_sha256": identity["varnet_ckpt_sha256"],
            "pmr_ckpt_path": CKPT_PMR_PATH,
            "pmr_ckpt_sha256": identity["pmr_ckpt_sha256"],
            "pmrp_ckpt_path": CKPT_PMRP_PATH,
            "pmrp_ckpt_sha256": identity["pmrp_ckpt_sha256"],
            "promptmr_official_commit": identity["pmrp_commit"],
            "fastmri_official_commit": identity["fastmri_commit"],
            "inherited_pipeline_script": INHERITED_SCRIPT,
            "inherited_pipeline_script_sha256": identity["inherited_script_sha256"],
            "inherited_pipeline_script2": INHERITED_SCRIPT_2,
            "inherited_pipeline_script2_sha256": identity["inherited_script2_sha256"],
            "g3_confirmation_json": G3_JSON_PATH,
        },
        "model_loads": models["loads"],
        "environment": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__,
            "lightning": __import__("lightning").__version__,
            "gpu_name": (torch.cuda.get_device_name(0)
                         if device.type == "cuda" else "cpu"),
            "interpreter": r"D:/anaconda3/envs/pytorch_cuda118/python.exe",
            "seed": SEED,
            "precision": "fp32; torch.inference_mode(); no autocast/quantization; "
                         "torch.backends.cudnn.benchmark=False",
        },
        "data": {
            "source_dir": DATA_DIR,
            "sample": ("REUSE of the frozen G3 confirmation 68-volume list "
                       "(within-sample method extension; not a new independent cohort); "
                       "official batch_0 AXT2, 16 slices, 768x396, coils in {12,14,20}, "
                       "rss (16,384,384); all 16 slices per volume"),
            "n_volumes": N_CONFIRM,
            "strata": {str(k): v for k, v in STRATUM_EXPECTED.items()},
            "slices_per_volume": EXPECTED_SLICES,
            "statistical_unit": "volume",
            "data_domain": ("official fastMRI brain multicoil validation batch_0; "
                            "validation-subset; all checkpoints' training may include "
                            "public validation data; never model-held-out, test, or "
                            "leaderboard"),
        },
        "conditions": {
            "C1": {"mask": "official EquispacedMaskFunc(center_fraction=0.08, acceleration=4)",
                   "use_seed": True, "seed": "tuple(map(ord, filename))",
                   "padding": "official metadata", "L_nom": C1["l_nom"],
                   "rho_to_L": {str(r): l for r, l in C1["grid"]},
                   "observed_sampled_cols_offsetNone": "all 68 filenames -> 123"},
            "C2": {"mask": "official EquispacedMaskFunc(center_fraction=0.04, acceleration=8)",
                   "use_seed": True, "seed": "tuple(map(ord, filename))",
                   "padding": "official metadata", "L_nom": C2["l_nom"],
                   "rho_to_L": {str(r): l for r, l in C2["grid"]},
                   "observed_sampled_cols_offsetNone": {
                       str(k): v for k, v in sorted(masks["c2_cols_observed"].items())}},
        },
        "intervention": {
            "only_change": ("num_low_frequencies fed to each model's official "
                            "sensitivity stage (VarNet sens_net / PromptMR sens_net); "
                            "rho in {0.25,0.50,0.75,1.00} x L_nom"),
            "shared_acquisition_layer": ("per (volume, condition): one sample object per "
                                         "model reused bitwise across all four rho; mask "
                                         "bitwise equal across models (asserted per slice); "
                                         "target and 320x320 crop bitwise equal across models "
                                         "(asserted per slice); official metric implementation "
                                         "shared; each model's own official spatial "
                                         "preprocessing (VN full 768 rows; PM height-crop to "
                                         "384 rows) is fixed across conditions and rho"),
            "static_trace": ("num_low_frequencies appears ONLY in the sens_net call of "
                             "all three model forwards (AST check in preflight)"),
        },
        "provenance": {
            "psnr_algorithm": ("fastmri.evaluate.psnr (skimage peak_signal_noise_ratio, "
                               "data_range=gt.max()) on (16,320,320) (target, pred) stacks; "
                               "one value per volume; PRIMARY metric"),
            "secondary_metrics": ("fastmri.evaluate nmse/ssim on the same stacks; "
                                  "JSON/appendix only; never enter Table 9"),
            "mean_algorithm": ("unweighted arithmetic mean of 68 volume-level values, "
                               "full-precision float64, no intermediate rounding"),
            "band_algorithm": ("studentized max-t (sup-t) simultaneous 95% band; B=10000; "
                               "seed=20260901; resampling unit=volume (the volume's whole "
                               "model x rho vector resampled jointly); a fresh "
                               "np.random.default_rng(20260901) instance per curve so each "
                               "curve is independently recomputable from the per-volume "
                               "values in this JSON; t*_b = max_rho |mean*_b(rho)-mean(rho)|/"
                               "max(SE*_b(rho),1e-12); q = np.percentile(t_b, 95) (linear "
                               "interpolation); FROZEN CALIBRATION SUGGESTION, not a "
                               "community standard; pointwise 1.96*SE is descriptive only"),
            "statistical_unit": "volume",
        },
        "phase1_smoke_executed_and_passed": True,
        "preflight_checks": {
            "identity_bindings": identity,
            "frozen_sample": {k: v for k, v in frozen.items() if k != "per_file"},
            "official_mask_measurements": {
                "c1_all_123": masks["c1_all_123"],
                "c2_all_nlf16": masks["c2_all_nlf16"],
                "c2_cols_observed_offsetNone": {
                    str(k): v for k, v in sorted(masks["c2_cols_observed"].items())}},
            "static_trace": traces, "acs_extractor_pads": pads,
            "mask_layer_equality": layer, "synthetic_forwards": synth},
        "per_condition_rows": [],
        "statistics": None,
        "gates": gates,
        "runtime": {},
        "proposed_but_unverified": [],
        "status": "IN_PROGRESS",
    }
    write_json_atomic(REPORT_PATH, report)

    t_sci = time.perf_counter()
    try:
        for vi, (fname, coils) in enumerate(CONFIRMATION_VOLUMES):
            path = os.path.join(DATA_DIR, fname)
            meta = official_metadata(path)
            g3_offset_c1 = frozen["per_file"][fname]["observed_mask_offset_C1"]
            with h5py.File(path, "r") as hf:
                num_slices = int(hf["kspace"].shape[0])
                assert num_slices == EXPECTED_SLICES, f"{fname}: {num_slices} slices"
                crops = {c: {m: {r: [] for r in RHO_GRID} for m in MODELS}
                         for c in ("C1", "C2")}
                targets = {c: [] for c in ("C1", "C2")}
                cond_masks = {}
                for cond_name in ("C1", "C2"):
                    cfg = CONDITIONS[cond_name]
                    lnom = cfg["l_nom"]
                    for s in range(EXPECTED_SLICES):
                        kspace = hf["kspace"][s]
                        target = hf[TARGET_KEY][s]
                        # --- VarNet sample (this object reused across all four rho) ---
                        vs = transforms_vn[cond_name](kspace, None, target, meta, fname, s)
                        cols, off = mask_cols_offset(vs.mask[0, 0, :, 0])
                        assert int(vs.num_low_frequencies) == lnom, \
                            f"{fname} {cond_name} s{s}: mask nlf {vs.num_low_frequencies}"
                        if cond_name == "C1":
                            assert cols == EXPECTED_MASK_COLS_C1, f"{fname} C1 s{s} cols {cols}"
                            assert off == g3_offset_c1, f"{fname} C1 s{s} offset {off}"
                        else:
                            ref = cond_masks.get("C2_ref")
                            if ref is not None:
                                assert (cols, off) == ref, f"{fname} C2 s{s} mask drift"
                        # --- PromptMR+ sample (official 5-slice clamped stack) ---
                        idx5 = [min(max(s + j, 0), num_slices - 1) for j in range(-2, 3)]
                        stacked5 = np.concatenate([hf["kspace"][i] for i in idx5], axis=0)
                        ps5 = transforms_pm[cond_name](stacked5, None, None, meta, fname, s)
                        # --- PromptMR v1 sample (official 3-slice clamped stack) ---
                        idx3 = [min(max(s + j, 0), num_slices - 1) for j in range(-1, 2)]
                        stacked3 = np.concatenate([hf["kspace"][i] for i in idx3], axis=0)
                        ps3 = transforms_pm[cond_name](stacked3, None, None, meta, fname, s)
                        assert int(ps5.num_low_frequencies) == lnom
                        assert int(ps3.num_low_frequencies) == lnom
                        # --- cross-model purity (every slice) ---
                        vn_mask_vec = vs.mask[0, 0, :, 0].cpu()
                        assert torch.equal(vn_mask_vec, ps5.mask[0, 0, :, 0].cpu()), \
                            f"{fname} {cond_name} s{s}: VN/PMRP mask mismatch"
                        assert torch.equal(vn_mask_vec, ps3.mask[0, 0, :, 0].cpu()), \
                            f"{fname} {cond_name} s{s}: VN/PMR mask mismatch"
                        vn_tcrop = T.center_crop(vs.target, EVAL_CROP).numpy().astype(np.float64)
                        p5_tcrop = T.center_crop(T.to_tensor(target), EVAL_CROP).numpy().astype(np.float64)
                        p3_tcrop = T.center_crop(T.to_tensor(target), EVAL_CROP).numpy().astype(np.float64)
                        assert np.array_equal(vn_tcrop, p5_tcrop) and np.array_equal(vn_tcrop, p3_tcrop), \
                            f"{fname} {cond_name} s{s}: target crop mismatch"
                        if s == 0:
                            cond_masks[cond_name] = (cols, off)
                            if cond_name == "C2":
                                cond_masks["C2_ref"] = (cols, off)
                        # --- forwards: ONLY the L argument varies with rho ---
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
                                crops[cond_name]["VN"][rho].append(
                                    T.center_crop(mag, EVAL_CROP)[0].cpu().numpy().astype(np.float64))
                                del kp, mag
                                o5 = pmrp(p5_mks, p5_mk, torch.tensor([L], device=device),
                                          [ps5.mask_type])
                                img5 = T.center_crop(PT.center_crop(o5["img_pred"], ps5.crop_size),
                                                     EVAL_CROP)
                                crops[cond_name]["PMRP"][rho].append(
                                    img5[0].cpu().numpy().astype(np.float64))
                                del o5, img5
                                o3 = pmr(p3_mks, p3_mk, torch.tensor([L], device=device),
                                         [ps3.mask_type])
                                img3 = T.center_crop(PT.center_crop(o3["img_pred"], ps3.crop_size),
                                                     EVAL_CROP)
                                crops[cond_name]["PMR"][rho].append(
                                    img3[0].cpu().numpy().astype(np.float64))
                                del o3, img3
                        del vn_mks, vn_mk, p5_mks, p5_mk, p3_mks, p3_mk, vs, ps5, ps3
                        targets[cond_name].append(vn_tcrop)
                    # --- volume-level official metrics for this condition ---
                    gt_stack = np.stack(targets[cond_name])
                    assert gt_stack.shape == (EXPECTED_SLICES,) + EVAL_CROP
                    for m in MODELS:
                        for rho, _L in CONDITIONS[cond_name]["grid"]:
                            pred = np.stack(crops[cond_name][m][rho])
                            assert pred.shape == gt_stack.shape
                            assert np.all(np.isfinite(pred)) and np.all(np.isfinite(gt_stack)), \
                                f"{fname} {cond_name} {m} rho={rho}: non-finite"
                            ms = metric_set(gt_stack, pred)
                            psnr_store[m][cond_name][rho].append(ms["PSNR"])
                            nmse_store[m][cond_name][rho].append(ms["NMSE"])
                            ssim_store[m][cond_name][rho].append(ms["SSIM"])
                    # --- qualitative capture (frozen three cases, slice 7) ---
                    if fname in QUAL_VOLTS:
                        key0 = fname[:-3]
                        qual[f"{key0}__{cond_name}__target"] = \
                            gt_stack[QUAL_SLICE_IDX].astype(np.float32)
                        for m in MODELS:
                            for rho, _L in CONDITIONS[cond_name]["grid"]:
                                qual[f"{key0}__{cond_name}__{m}__rho_{rho}"] = \
                                    np.stack(crops[cond_name][m][rho])[QUAL_SLICE_IDX].astype(np.float32)
                    del crops[cond_name], targets[cond_name]
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                # --- per-condition row ---
                row = {"volume": fname, "coils": coils, "cells": []}
                for cond_name in ("C1", "C2"):
                    cols, off = cond_masks[cond_name]
                    for m in MODELS:
                        for rho, L in CONDITIONS[cond_name]["grid"]:
                            row["cells"].append({
                                "filename": fname, "coil": coils, "condition": cond_name,
                                "rho": rho, "L": L, "model": m,
                                "PSNR": psnr_store[m][cond_name][rho][vi],
                                "NMSE": nmse_store[m][cond_name][rho][vi],
                                "SSIM": ssim_store[m][cond_name][rho][vi]})
                    row[f"{cond_name}_mask"] = {"sampled_cols": cols, "offset": off,
                                                "num_low_frequencies": CONDITIONS[cond_name]["l_nom"]}
                # --- G3 reference consistency (descriptive only) ---
                g3r = g3_rows[fname]
                row["g3_reference_abs_psnr_diff_descriptive"] = {
                    "VN_C1_rho1.00_vs_G3_ACS32": abs(psnr_store["VN"]["C1"][1.00][vi]
                                                     - g3r["VarNet"]["ACS32"]["PSNR"]),
                    "VN_C1_rho0.50_vs_G3_ACS16": abs(psnr_store["VN"]["C1"][0.50][vi]
                                                     - g3r["VarNet"]["ACS16"]["PSNR"]),
                    "PMRP_C1_rho1.00_vs_G3_ACS32": abs(psnr_store["PMRP"]["C1"][1.00][vi]
                                                       - g3r["PromptMRPlus"]["ACS32"]["PSNR"]),
                    "PMRP_C1_rho0.50_vs_G3_ACS16": abs(psnr_store["PMRP"]["C1"][0.50][vi]
                                                       - g3r["PromptMRPlus"]["ACS16"]["PSNR"]),
                }
                rows.append(row)
                print(f"[{vi+1}/{N_CONFIRM}] {fname} (c={coils}) "
                      f"VN_C1_32={psnr_store['VN']['C1'][1.00][vi]:.4f} "
                      f"VN_C2_16={psnr_store['VN']['C2'][1.00][vi]:.4f} "
                      f"PMR_C1_32={psnr_store['PMR']['C1'][1.00][vi]:.4f} "
                      f"PMRP_C1_32={psnr_store['PMRP']['C1'][1.00][vi]:.4f}", flush=True)
            # --- incremental atomic checkpoint ---
            report["per_condition_rows"] = rows
            report["runtime"] = {"science_wall_seconds": time.perf_counter() - t_sci,
                                 "volumes_done": vi + 1}
            write_json_atomic(REPORT_PATH, report)
            # --- budget envelope guard ---
            if vi >= 5:
                elapsed = time.perf_counter() - t_sci
                avg = elapsed / (vi + 1)
                projected = avg * N_CONFIRM
                if projected > SCIENCE_WALL_BUDGET_S:
                    report["status"] = "STOPPED_BUDGET_ENVELOPE"
                    report["runtime"]["projected_science_seconds"] = projected
                    write_json_atomic(REPORT_PATH, report)
                    print(f"[STOP] budget envelope: projected science time "
                          f"{projected / 3600:.2f} h > 7.7 h. No silent expansion.", flush=True)
                    return 1
    except BaseException as exc:  # crash: record and stop; no repair, no rerun
        report["status"] = f"STOPPED_CRASH::{type(exc).__name__}:{exc}"
        report["runtime"] = {"science_wall_seconds": time.perf_counter() - t_sci,
                             "volumes_done": len(rows)}
        write_json_atomic(REPORT_PATH, report)
        print(f"[STOP] science run crashed: {type(exc).__name__}: {exc}", flush=True)
        raise

    # ======================================================================== #
    # frozen statistics + mechanical Table 9
    # ======================================================================== #
    finite_all = all(np.isfinite(psnr_store[m][c][r]).all()
                     for m in MODELS for c in ("C1", "C2") for r in RHO_GRID)
    complete = bool(len(rows) == N_CONFIRM and finite_all
                    and all(gates.get(k) for k in
                            ("identity_bindings", "frozen_sample_metadata", "strict_loads",
                             "n_buffer_shim_consistent", "static_trace_sens_stage_only",
                             "acs_extractor_pads", "official_mask_measurements",
                             "mask_layer_equality", "synthetic_forwards")))

    # Q_{m,c}(rho): unweighted mean over the 68 volume-level PSNRs
    q = {m: {c: {r: float(np.mean(psnr_store[m][c][r])) for r in RHO_GRID}
             for c in ("C1", "C2")} for m in MODELS}

    # D curves: 3 pairs x 2 conditions
    d_curves = {}
    for a, b in PAIRS:
        for c in ("C1", "C2"):
            mat = np.stack([np.array([psnr_store[a][c][r][i] - psnr_store[b][c][r][i]
                                      for r in RHO_GRID])
                            for i in range(N_CONFIRM)])  # (68, 4)
            band = simul_band(mat)
            label = classify_curve(band["dhat"], band["lo"], band["hi"])
            key = f"{a}_vs_{b}__{c}"
            d_curves[key] = {
                "pair": [a, b], "condition": c,
                "sign_convention": "positive = first model better",
                "rho_grid": list(RHO_GRID),
                "D_point": {str(r): float(band["dhat"][j]) for j, r in enumerate(RHO_GRID)},
                "SE": {str(r): float(band["se"][j]) for j, r in enumerate(RHO_GRID)},
                "simultaneous_band_95": {str(r): {"lo": float(band["lo"][j]),
                                                  "hi": float(band["hi"][j])}
                                         for j, r in enumerate(RHO_GRID)},
                "pointwise_196_descriptive_only": {
                    str(r): {"lo": float(band["pointwise_lo"][j]),
                             "hi": float(band["pointwise_hi"][j])}
                    for j, r in enumerate(RHO_GRID)},
                "q_maxt": band["q"],
                "I_descriptive": {str(r): float(band["dhat"][j] - band["dhat"][RHO_GRID.index(1.00)])
                                  for j, r in enumerate(RHO_GRID)},
                "label": label,
            }
            strata_desc = {}
            for coil in sorted(STRATUM_EXPECTED):
                idxs = [i for i, (_f, cc) in enumerate(CONFIRMATION_VOLUMES) if cc == coil]
                strata_desc[str(coil)] = {
                    "n": len(idxs),
                    "D_point": {str(r): float(mat[idxs, j].mean())
                                for j, r in enumerate(RHO_GRID)}}
            d_curves[key]["stratum_descriptive"] = strata_desc

    # Delta curves: 3 models x 2 conditions (reference rho = 1.00)
    delta_curves = {}
    for m in MODELS:
        for c in ("C1", "C2"):
            rho_nr = [r for r in RHO_GRID if r != 1.00]
            mat = np.stack([np.array([psnr_store[m][c][r][i] - psnr_store[m][c][1.00][i]
                                      for r in rho_nr])
                            for i in range(N_CONFIRM)])  # (68, 3)
            band = simul_band(mat)
            events = {str(r): bool(band["lo"][j] > 0 or band["hi"][j] < 0)
                      for j, r in enumerate(rho_nr)}
            delta_curves[f"{m}__{c}"] = {
                "model": m, "condition": c, "reference_rho": 1.00, "rho_grid": rho_nr,
                "Delta_point": {str(r): float(band["dhat"][j]) for j, r in enumerate(rho_nr)},
                "SE": {str(r): float(band["se"][j]) for j, r in enumerate(rho_nr)},
                "simultaneous_band_95": {str(r): {"lo": float(band["lo"][j]),
                                                  "hi": float(band["hi"][j])}
                                         for j, r in enumerate(rho_nr)},
                "q_maxt": band["q"],
                "band_excludes_0": events,
            }

    # ------------------------------------------------------------------ #
    # Table 9 (mechanical; preregistration section 6; no rule changes)
    # ------------------------------------------------------------------ #
    def _excl0(key, rho):
        bb = d_curves[key]["simultaneous_band_95"][str(rho)]
        return bool(bb["lo"] > 0 or bb["hi"] < 0)

    def _label(key):
        return d_curves[key]["label"]

    g2_c1 = any(_label(f"{a}_vs_{b}__C1") in ("DOMINANCE", "STRICT_CROSSING")
                for a, b in PAIRS)
    g2_c2 = any(_label(f"{a}_vs_{b}__C2") in ("DOMINANCE", "STRICT_CROSSING")
                for a, b in PAIRS)
    pairs_hit = [f"{a}_vs_{b}" for a, b in PAIRS
                 if _label(f"{a}_vs_{b}__C1") in ("DOMINANCE", "STRICT_CROSSING")
                 or _label(f"{a}_vs_{b}__C2") in ("DOMINANCE", "STRICT_CROSSING")]
    g3 = len(pairs_hit) >= 2
    c1_new_rho = any(
        _label(f"{a}_vs_{b}__C1") in ("DOMINANCE", "STRICT_CROSSING")
        and (_excl0(f"{a}_vs_{b}__C1", 0.25) or _excl0(f"{a}_vs_{b}__C1", 0.75))
        for a, b in PAIRS)
    g4 = bool(g2_c2 or c1_new_rho)
    g1 = complete
    generalized = bool(g1 and g2_c1 and g2_c2 and g3 and g4)
    any_dom_cross = any(v["label"] in ("DOMINANCE", "STRICT_CROSSING")
                        for v in d_curves.values())
    delta_events = {k: {r: e for r, e in v["band_excludes_0"].items() if e}
                    for k, v in delta_curves.items() if any(v["band_excludes_0"].values())}
    partial = bool((not generalized) and complete and (any_dom_cross or bool(delta_events)))
    existence = bool(complete and (not generalized) and (not partial))
    if not complete:
        verdict = "INVALID"
    elif generalized:
        verdict = "GENERALIZED_METHOD_SUPPORT"
    elif partial:
        verdict = "PARTIAL_METHOD_SUPPORT"
    else:
        verdict = "EXISTENCE_ONLY"
    assert (generalized + partial + existence) <= 1

    table9 = {
        "G1_matrix_complete": bool(g1),
        "G2_C1_any_pair_dom_or_cross": bool(g2_c1),
        "G2_C2_any_pair_dom_or_cross": bool(g2_c2),
        "G3_at_least_two_pairs_anywhere": bool(g3),
        "G3_pairs": pairs_hit,
        "G4_not_only_seen_flip": bool(g4),
        "G4_detail": {"G2_C2_independent": bool(g2_c2),
                      "C1_uses_new_rho_0.25_or_0.75_band_excl_0": bool(c1_new_rho),
                      "per_pair_C1_excl0_at_new_rho": {
                          f"{a}_vs_{b}": {"rho_0.25": _excl0(f"{a}_vs_{b}__C1", 0.25),
                                          "rho_0.75": _excl0(f"{a}_vs_{b}__C1", 0.75)}
                          for a, b in PAIRS}},
        "delta_events_rho_ne_1": delta_events,
        "verdict": verdict,
        "rule_source": ("Q2_SSRE_PREREGISTRATION_001.md section 6 (frozen); mechanical "
                        "order GENERALIZED -> PARTIAL -> EXISTENCE_ONLY; INVALID on any "
                        "gate failure or review FAIL"),
    }

    ref_keys = list(rows[0]["g3_reference_abs_psnr_diff_descriptive"].keys())
    ref_summary = {k: {"max_abs_diff": float(max(r["g3_reference_abs_psnr_diff_descriptive"][k]
                                                 for r in rows)),
                       "mean_abs_diff": float(np.mean([r["g3_reference_abs_psnr_diff_descriptive"][k]
                                                       for r in rows]))}
                   for k in ref_keys}

    report["per_condition_rows"] = rows
    report["statistics"] = {
        "Q": {m: {c: {str(r): q[m][c][r] for r in RHO_GRID} for c in ("C1", "C2")}
              for m in MODELS},
        "d_curves": d_curves,
        "delta_curves": delta_curves,
        "table9": table9,
        "curve_labels": {k: v["label"] for k, v in d_curves.items()},
        "g3_reference_consistency_descriptive_only": {
            "note": ("recomputed C1 rho=1.00/0.50 VN+PMRP cells vs the frozen G3 "
                     "confirmation PSNRs; descriptive only; not a gate; no G3 number "
                     "enters this matrix's statistics"),
            **ref_summary},
    }
    report["gates"]["run_complete_finite"] = bool(len(rows) == N_CONFIRM and finite_all)
    report["gates"]["all_pass"] = bool(all(v for v in report["gates"].values()
                                           if v is not None))
    report["qualitative"] = {"npz_path": QUAL_NPZ_PATH, "volumes": list(QUAL_VOLTS),
                             "slice_idx_0based": QUAL_SLICE_IDX,
                             "content": "target + 3 models x 2 conditions x 4 rho "
                                        "magnitude crops (320x320 float32)"}
    report["runtime"] = {"wall_seconds_total": time.perf_counter() - t_start,
                         "science_wall_seconds": time.perf_counter() - t_sci,
                         "volumes_done": len(rows)}
    report["proposed_but_unverified"] = [
        "the D-curve labels and the Table 9 verdict are the scientific results of this "
        "run, not validity checks",
        "results cover only these 68 reused validation-subset volumes under C1/C2 and "
        "the four prespecified support points; no population, cross-dataset, "
        "cross-contrast or clinical generalization is licensed",
        "the studentized simultaneous 95% band is a frozen calibration suggestion for "
        "this experiment, not a community standard",
        "SSRE novelty for reviewers remains NOVELTY UNRESOLVED",
        ("the mechanism of any model's support sensitivity is NOT identified (out of "
         "scope; no gauge or mechanism line was run)"),
    ]
    report["status"] = verdict if report["gates"]["all_pass"] else "EXECUTION_FAIL"
    write_json_atomic(REPORT_PATH, report)
    os.makedirs(os.path.dirname(QUAL_NPZ_PATH), exist_ok=True)
    np.savez_compressed(QUAL_NPZ_PATH, **qual)

    print("=" * 78)
    for name, ok in report["gates"].items():
        print(f"  {'OK ' if ok else 'XX '}{name}")
    for m in MODELS:
        for c in ("C1", "C2"):
            print(f"  Q({m},{c}): " + " ".join(f"{r}={q[m][c][r]:.4f}" for r in RHO_GRID))
    for key, v in d_curves.items():
        print(f"  {key}: label={v['label']} "
              + " ".join(f"D({r})={v['D_point'][r]:+.4f}" for r in map(str, RHO_GRID)))
    print(f"  TABLE 9 VERDICT: {verdict}")
    print(f"  report -> {REPORT_PATH}")
    print("=" * 78, flush=True)
    return 0 if report["gates"]["all_pass"] else 1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "preflight", "science"), required=True)
    args = parser.parse_args()
    t_start = time.perf_counter()
    script_sha = sha256_of(os.path.abspath(__file__))
    print(f"[mode] {args.mode}; script sha256: {script_sha}", flush=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False

    if args.mode == "science" and os.path.exists(REPORT_PATH):
        print(f"[STOP] report path already exists: {REPORT_PATH}. No rerun.", flush=True)
        return 1

    gates, models, payload = run_all_gates(need_models=True)
    if models is None:
        return 1
    if args.mode == "smoke":
        ok = run_smoke(models)
        print("[smoke done -- nothing written to any science JSON]", flush=True)
        return 0 if ok else 1
    if args.mode == "preflight":
        print("[preflight PASS -- frozen script ready for the single science run]", flush=True)
        return 0
    return run_science(models, gates, payload, t_start=t_start)


if __name__ == "__main__":
    sys.exit(main())
