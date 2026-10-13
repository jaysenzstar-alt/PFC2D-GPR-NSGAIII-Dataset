"""Prepare native PFC cases or extract the twelve supplied histories.

Analyze archived histories:
  python direct_validation/validation.py analyze --output reproduced/direct

Prepare fresh working directories using a compatible licensed plug-in:
  python direct_validation/validation.py prepare-runs --output reproduced/runs
      --plugin PATH_TO_PLUGIN/fishuser_function005_64.dll

In each case directory, use PFC2D 5.00 Release 025 to call launch.p2dat.
Then analyze with --histories reproduced/runs and a new --output directory.
Preparing directories does not launch PFC. Runtime components are not included.
Response definitions are in manifest.json. Input and output locations are
described in REPRODUCE.md.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent
PARAMS = ['pb_emod_gpa', 'pb_kra', 'pb_ten_mpa', 'pb_coh_mpa', 'pb_fa_deg']
OUTPUTS = ['peak_stress_mpa', 'peak_strain_permille', 'elastic_modulus_gpa', 'poisson_ratio']

def cases():
    with (ROOT/'model_files/representative_candidates_nsgaIII.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if [int(c['sample']) for c in rows] != list(range(1, 13)):
        raise ValueError('Expected twelve parameter rows ordered by sample 1-12.')
    return rows

def metrics(path):
    rows = []
    for line in path.read_text(encoding='ascii').splitlines():
        parts = line.replace(',', ' ').split()
        if len(parts) == 8:
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                pass
    a = np.asarray(rows)
    assert len(a) > 10 and np.isfinite(a).all() and np.all(np.diff(a[:, 0]) > 0)
    peak = int(np.argmax(a[:, 1]))
    assert 0 < peak < len(a)-1
    assert a[-1, 1] <= 0.12*a[peak, 1], 'History does not include post-peak decay'
    result = dict(peak_stress_mpa=float(a[peak, 1]), peak_strain_permille=float(a[peak, 2]*1000), rows=len(a))
    readouts = {}
    for fraction in [0.5, 0.25, 0.4, 0.6]:
        i = int(np.argmin(abs(a[:peak+1, 1]-fraction*a[peak, 1])))
        assert a[i, 2] > 0 and abs(a[i, 5]) > 0
        assert math.isclose(a[i, 4], abs(0.001*a[i, 1]/a[i, 2]), rel_tol=1e-7)
        assert math.isclose(a[i, 3], abs(a[i, 6]/a[i, 5]), rel_tol=1e-7)
        readouts[str(fraction)] = dict(step=int(a[i, 0]), stress_fraction=float(a[i, 1]/a[peak, 1]),
                                      elastic_modulus_gpa=float(a[i, 4]), poisson_ratio=float(a[i, 3]))
    result.update(elastic_modulus_gpa=readouts['0.5']['elastic_modulus_gpa'],
                  poisson_ratio=readouts['0.5']['poisson_ratio'], readouts=readouts)
    return result

def analyze(history_root, output):
    rows = []
    for case in cases():
        sample = int(case['sample'])
        result = dict(sample=sample, parameters={k: float(case[k]) for k in PARAMS},
                      **metrics(history_root/f'current_{sample:02d}'/'history.his'))
        errors = []
        for key in OUTPUTS:
            target = float(case['target_'+key]); pred = float(case['pred_'+key]); direct = result[key]
            result['target_'+key] = target; result['gpr_'+key] = pred
            error = abs(direct-target)/abs(target); errors.append(error)
            result['pfc_target_error_'+key] = error
            result['gpr_target_error_'+key] = abs(pred-target)/abs(target)
            result['pfc_gpr_gap_'+key] = abs(direct-pred)/abs(target)
        result['mare'] = sum(errors)/4; result['maximum_relative_error'] = max(errors)
        rows.append(result)
    output.mkdir(parents=True, exist_ok=False)
    (output/'joint_responses.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    flat = [{k: v for k, v in row.items() if not isinstance(v, dict)} for row in rows]
    with (output/'target_gpr_pfc_comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(f'Extracted {len(rows)} cases; mean MARE = {100*np.mean([r["mare"] for r in rows]):.6f}%')

def prepare_runs(destination, plugin):
    if not plugin.is_file():
        raise FileNotFoundError('Provide the compatible fishuser_function005_64.dll from your licensed installation.')
    if destination.exists():
        raise FileExistsError('Choose a new directory to preserve existing runs.')
    parameter_rows = cases()
    destination.mkdir(parents=True, exist_ok=False)
    for case in parameter_rows:
        sample = int(case['sample']); work = destination/f'current_{sample:02d}'; work.mkdir()
        shutil.copy2(ROOT/'model_files/yuya.p2sav', work/'g_1.p2sav')
        for name in ['3-bonded.p2dat', '44-ucs.p2dat', 'fracture.fis']:
            shutil.copy2(ROOT/'model_files'/name, work/name)
        shutil.copy2(plugin, work/'fishuser_function005_64.dll')
        shutil.copy2(ROOT/'drivers'/f'current_{sample:02d}.p2dat', work/'launch.p2dat')
    print('Prepared twelve case directories. No simulations have been launched.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    analysis = actions.add_parser('analyze')
    analysis.add_argument('--histories', type=Path, default=ROOT/'recorded_histories')
    analysis.add_argument('--output', type=Path, required=True)
    preparation = actions.add_parser('prepare-runs')
    preparation.add_argument('--output', type=Path, required=True)
    preparation.add_argument('--plugin', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'analyze': analyze(args.histories.resolve(), args.output.resolve())
    else: prepare_runs(args.output.resolve(), args.plugin.resolve())
