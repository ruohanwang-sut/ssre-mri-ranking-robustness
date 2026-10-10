"""Reproduce the published SSRE bands from frozen volume metrics, without MRI inference.

Statistical functions below are extracted unchanged from the frozen analysis script.
This entry point is a publication-time packaging addition, not a new experiment.
"""
from pathlib import Path
import csv,json
import numpy as np
BOOTSTRAP_B=10000
BOOTSTRAP_SEED=20260901
SE_FLOOR=1e-12
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

def main():
    root=Path(__file__).resolve().parents[1]
    with (root/'data/volume_metrics.csv').open(newline='',encoding='utf-8') as f:
        cells=list(csv.DictReader(f))
    with (root/'data/volumes.csv').open(newline='',encoding='utf-8') as f:
        volumes=[r['volume'] for r in csv.DictReader(f)]
    frozen=json.loads((root/'data/frozen_statistics.json').read_text())['d_curves']
    lookup={(r['filename'],r['model'],r['condition'],float(r['rho'])):float(r['PSNR']) for r in cells}
    assert len(cells)==1632 and len(lookup)==1632 and len(volumes)==68
    worst=0.0
    for key,v in frozen.items():
        a,b=v['pair'];c=v['condition'];grid=v['rho_grid']
        matrix=np.array([[lookup[(f,a,c,r)]-lookup[(f,b,c,r)] for r in grid] for f in volumes])
        band=simul_band(matrix)
        label=classify_curve(band['dhat'],band['lo'],band['hi'])
        assert label==v['label'],(key,label,v['label'])
        for field,source in [('dhat','D_point'),('se','SE')]:
            worst=max(worst,float(np.max(np.abs(band[field]-np.array([v[source][str(r)] for r in grid])))))
        for field in ('lo','hi'):
            worst=max(worst,float(np.max(np.abs(band[field]-np.array([v['simultaneous_band_95'][str(r)][field] for r in grid])))))
        worst=max(worst,abs(band['q']-v['q_maxt']))
        print(key,label)
    assert worst<=1e-12,f'Maximum numerical deviation: {worst}'
    print(f'PASS: 1632 cells, 68 volumes, six classifications; maximum deviation {worst:.17g}')

if __name__=='__main__':
    main()
