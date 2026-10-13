"""Recompute approximate errors from displayed historical responses, not native runs.

Usage: python recompute_historical_errors.py OUTPUT.csv
The historical reported MARE is preserved, never replaced by the approximate mean.
"""
from pathlib import Path
import csv
import sys

source=Path(__file__).resolve().parent/'table9_historical_direct_pfc.csv'
output=Path(sys.argv[1]) if len(sys.argv)>1 else source.with_name('tableS15_recomputed.csv')
responses=['peak_stress_mpa','peak_strain_permille','elastic_modulus_gpa','poisson_ratio']
rows=[]
with source.open(encoding='utf-8-sig',newline='') as f:
    for raw in csv.DictReader(f):
        row={'sample':int(raw['sample'])}
        values=[]
        for key in responses:
            target=float(raw['target_'+key]);actual=float(raw['historical_pfc_'+key])
            assert target!=0
            error=abs(actual-target)/abs(target)
            row[key+'_approx_error']=error;values.append(error)
        row['approx_mean_error']=sum(values)/4
        row['approx_max_error']=max(values)
        row['original_reported_mare']=float(raw['reported_mare'])
        row['calculation_basis']='ROUNDED_DISPLAYED_RESPONSES_NOT_NATIVE_FULL_PRECISION'
        rows.append(row)
assert len(rows)==12
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f'Wrote {len(rows)} historical cases to {output}')
