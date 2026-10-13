"""Reaggregate archived metrics and stored NSGA-III populations.

The inputs are the reported frozen records, not a newly trained pipeline.
Use --output to choose a new directory outside surrogate_analysis.
Use --tables-only to skip figure exports. No PFC or MATLAB runs are started.
"""
from pathlib import Path
import argparse
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = None
P = ['pb_emod_gpa','pb_kra','pb_ten_mpa','pb_coh_mpa','pb_fa_deg']
LABELS = ['Bond modulus','Stiffness ratio','Tensile strength','Cohesion','Friction angle']
LOW = np.array([15.,1.,15.,15.,15.])
SPAN = np.array([60.,2.,80.,20.,60.])
RESP = ['peak_stress','peak_strain','elastic_modulus','poisson_ratio']
ERRORS = ['err_'+r for r in RESP]
BLUE, ORANGE = '#0072B2', '#D55E00'

def csv(frame, name):
    frame.to_csv(OUT/name,index=False,float_format='%.17g')

def savefig(fig, name):
    dest = OUT / 'figures'
    dest.mkdir(exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(dest / f'{name}.{ext}', dpi=600, facecolor='white')
    with Image.open(dest / f'{name}.png') as im:
        im.convert('RGB').save(dest / f'{name}.tif', compression='tiff_lzw', dpi=(600, 600))
    plt.close(fig)

def main(output_dir: Path, tables_only: bool = False):
    global OUT
    OUT = output_dir.resolve()
    if OUT == ROOT or ROOT in OUT.parents:
        raise ValueError('Choose an output directory outside surrogate_analysis.')
    OUT.mkdir(parents=True, exist_ok=False)
    inputs=sorted((ROOT/'modeling_results/optimizer').glob('pareto_sample*_NSGAIII_run*.csv'))
    assert len(inputs)==120
    records=[]
    for path in inputs:
        frame=pd.read_csv(path)
        frame['source_file']=path.relative_to(ROOT).as_posix()
        frame['source_row_1based']=np.arange(1,len(frame)+1)
        records.append(frame)
    pop=pd.concat(records,ignore_index=True)
    assert len(pop)==7107 and not pop[P+ERRORS+['mean_relative_error']].isna().any().any()
    assert np.allclose(pop[ERRORS].mean(axis=1),pop.mean_relative_error,atol=1e-12)
    best=pop.groupby('sample').mean_relative_error.transform('min')
    rank_rows=[]; target_rows=[]; component_rows=[]; count_rows=[]
    for delta in [.0025,.005,.01]:
        kept=pop.loc[pop.mean_relative_error<=best+delta].copy()
        unique=kept.drop_duplicates(['sample']+P).copy()
        quant=unique.groupby('sample')[P].quantile([.25,.75]).unstack()
        iqr=(unique.groupby('sample')[P].quantile(.75)-unique.groupby('sample')[P].quantile(.25))/SPAN
        nrng=(unique.groupby('sample')[P].max()-unique.groupby('sample')[P].min())/SPAN
        for j,param in enumerate(P):
            rank_rows.append({'delta':delta,'parameter':param,'unique_vectors':len(unique),
                              'mean_normalized_iqr':iqr[param].mean(),'median_normalized_iqr':iqr[param].median(),
                              'mean_normalized_range':nrng[param].mean()})
        for sample,g in kept.groupby('sample'):
            u=unique.loc[unique['sample']==sample]
            count_rows.append({'delta':delta,'sample':int(sample),'global_best_mean_error':float(best.loc[g.index].iloc[0]),
                               'common_mean_error_limit':float(best.loc[g.index].iloc[0]+delta),
                               'retained_rows_before_dedup':len(g),'unique_vectors':len(u),'contributing_runs':g.run.nunique(),
                               'run_ids':';'.join(map(str,sorted(g.run.unique())))})
            for j,param in enumerate(P):
                target_rows.append({'delta':delta,'sample':int(sample),'parameter':param,'n':len(u),
                                    'normalized_iqr':iqr.loc[sample,param],'normalized_range':nrng.loc[sample,param]})
            for err in ERRORS:
                component_rows.append({'delta':delta,'sample':int(sample),'component':err,
                                       'median':u[err].median(),'p95':u[err].quantile(.95),'maximum':u[err].max()})
        if delta==.005:
            csv(kept,'retained_occurrences_delta005.csv')
            provenance=kept.groupby(['sample']+P,sort=False).agg(
                source_occurrences=('source_file','size'),
                source_runs=('run',lambda x:';'.join(map(str,sorted(set(x))))),
                source_locations=('source_file',lambda x:';'.join(x))).reset_index()
            csv(unique.merge(provenance,on=['sample']+P,validate='one_to_one'),'retained_unique_candidates_delta005.csv')
            csv(kept.groupby(['sample','run']).size().rename('retained_occurrences').reset_index(),'retained_run_counts_delta005.csv')
            main_unique=unique
    ranking=pd.DataFrame(rank_rows)
    ranking['rank']=ranking.groupby('delta').mean_normalized_iqr.rank(ascending=False,method='min').astype(int)
    csv(ranking.sort_values(['delta','rank']),'dispersion_sensitivity.csv')
    csv(pd.DataFrame(target_rows),'dispersion_by_target.csv')
    csv(pd.DataFrame(count_rows),'candidate_counts_by_target.csv')
    csv(pd.DataFrame(component_rows),'candidate_component_error_summary.csv')
    csv(pop.groupby(['sample','run']).mean_relative_error.min().rename('run_best_mean_error').reset_index(),'run_best_errors.csv')

    legacy=ROOT/'model_diagnostics/results'
    inv=pd.read_csv(legacy/'inverse_prediction/inverse_baseline_runs.csv')
    inv['mse']=inv.rmse**2
    csv(inv,'inverse_run_metrics_with_mse.csv')
    sums=inv.groupby(['model','parameter'])[['mse','rmse','mare','r2']].agg(['mean','std'])
    sums.columns=['_'.join(c) for c in sums.columns]
    sums=sums.reset_index();sums['n']=10
    csv(sums,'table6_inverse_summary.csv')
    learn=pd.read_csv(legacy/'sample_size/sample_size_repeated_runs.csv')
    curve=learn.groupby(['training_size','response'])[['rmse','mare','r2']].agg(['mean','std'])
    curve.columns=['_'.join(c) for c in curve.columns]
    curve=curve.reset_index();curve['n']=5;curve['n_test']=625
    csv(curve,'fig7_learning_curve_summary.csv')

    if not tables_only:
        with plt.rc_context({'font.family':'Times New Roman','font.size':10,'axes.linewidth':.8,'pdf.fonttype':42,'svg.fonttype':'none'}):
            fig,ax=plt.subplots(figsize=(6.8,3.7),layout='constrained')
            for model,col,marker,offset in [('inverse_GPR',BLUE,'o',-.09),('inverse_NN',ORANGE,'s',.09)]:
                s=sums.loc[sums.model==model].set_index('parameter').loc[P]
                ax.errorbar(np.arange(5)+offset,s.r2_mean,yerr=s.r2_std,fmt=marker,color=col,capsize=3,
                            label='GPR' if model=='inverse_GPR' else 'NN (fitrnet)')
            ax.set(xticks=range(5),xticklabels=['Bond\nmodulus','Stiffness\nratio','Tensile\nstrength','Cohesion','Friction\nangle'],ylabel='Held-out $R^2$',ylim=(0,1.07))
            ax.legend(frameon=False,loc='lower left');ax.grid(axis='y',alpha=.25)
            savefig(fig,'Fig06_inverse_diagnostic')
            fig,axes=plt.subplots(2,2,figsize=(6.8,5.4),layout='constrained')
            keys=['peak_stress_mpa','poisson_ratio','elastic_modulus_gpa','peak_strain_permille']
            titles=['Peak stress','Poisson\u2019s ratio','Elastic modulus','Peak strain']
            units=['RMSE (MPa)','RMSE','RMSE (GPa)','RMSE (\u2030)']
            for idx,(ax,key,title,unit) in enumerate(zip(axes.flat,keys,titles,units)):
                s=curve.loc[curve.response==key].sort_values('training_size')
                ax.errorbar(s.training_size,s.rmse_mean,yerr=s.rmse_std,fmt='o-',color=BLUE,capsize=3,lw=1)
                ax.set(xlabel='Training cases',ylabel=unit,xticks=[400,800,1200,1600,2000,2500])
                ax.set_title(f'({chr(97+idx)}) {title}',loc='left',fontsize=11)
                ax.tick_params(axis='x',labelsize=9);ax.grid(alpha=.25)
            savefig(fig,'Fig07_archived_fixed_test_learning_curve')
            summary=pd.read_csv(ROOT/'modeling_results/optimizer/nsga_comparison_summary.csv')
            fig,axes=plt.subplots(2,3,figsize=(7.48,5.51),layout='constrained')
            for alg,col,marker in [('NSGAII',BLUE,'o'),('NSGAIII',ORANGE,'^')]:
                frame=pd.read_csv(ROOT/f'modeling_results/optimizer/pareto_sample01_{alg}_run01.csv')
                label='NSGA-II' if alg=='NSGAII' else 'NSGA-III'
                axes[0,0].scatter(frame.err_peak_stress,frame.err_peak_strain,s=12,marker=marker,color=col,label=label,alpha=.75)
                axes[0,1].scatter(frame.err_elastic_modulus,frame.err_poisson_ratio,s=12,marker=marker,color=col,alpha=.75)
                part=summary.loc[summary.algorithm==alg].sort_values('sample')
                x=part['sample']+(-.12 if alg=='NSGAII' else .12)
                for ax,metric in [(axes[0,2],'best_error'),(axes[1,0],'igd'),(axes[1,1],'hypervolume')]:
                    y=part['mean_'+metric] if metric!='hypervolume' else 1-part['mean_'+metric]
                    ax.errorbar(x,y,yerr=part['std_'+metric],fmt=marker,color=col,ms=3,capsize=2,lw=.8)
                    ax.set(xticks=range(1,13),xlabel='Target');ax.tick_params(labelsize=8)
            axes[0,0].set(xlabel='Peak-stress relative error',ylabel='Peak-strain relative error');axes[0,0].legend(frameon=False,fontsize=8)
            axes[0,1].set(xlabel='Modulus relative error',ylabel='Poisson-ratio relative error')
            for ax,label in [(axes[0,2],'Best mean relative error'),(axes[1,0],'IGD'),(axes[1,1],'Hypervolume gap (1 - HV)')]:ax.set_ylabel(label)
            s=ranking.loc[ranking.delta==.005].sort_values('mean_normalized_iqr')
            axes[1,2].barh([dict(zip(P,LABELS))[p] for p in s.parameter],s.mean_normalized_iqr,color='#666666')
            axes[1,2].set_xlabel('Mean normalized IQR');axes[1,2].tick_params(labelsize=8)
            for i,ax in enumerate(axes.flat):
                ax.set_title(f'({chr(97+i)})',loc='left',fontsize=11)
                ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
            savefig(fig,'FigS01_optimizer_and_target_conditioned_dispersion')

    targets=pd.read_csv(ROOT/'input_data/targets_12.csv')
    recovered=pd.read_csv(ROOT/'target_source_records/target_workbook_provenance.csv')
    assert len(recovered)==12 and recovered['sample'].is_unique
    wb=ROOT/recovered.source_workbook.iloc[0]
    assert wb.is_file()
    recovered=recovered.set_index('sample').loc[targets['sample']].reset_index()
    response_fields=['target_peak_stress_mpa','target_peak_strain_permille','target_elastic_modulus_gpa','target_poisson_ratio']
    assert np.allclose(targets[response_fields],recovered[response_fields],rtol=0,atol=1e-8)
    csv(recovered,'target_workbook_provenance.csv')
    summary_out={'analysis_version':'R3-publication','population_files':len(inputs),'population_rows':len(pop),
                 'retained_unique_delta005':len(main_unique),'component_maxima_delta005':main_unique[ERRORS].max().to_dict(),
                 'archived_inverse_rows':len(inv),'archived_learning_curve_rows':len(learn),
                 'PFC_simulations_run':0,'MATLAB_fits_run':0,'python':platform.python_version(),
                 'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__,
                 'scope':'Stored-population reanalysis and archived-run aggregation; direct simulations are analysed separately in direct_validation.'}
    (OUT/'analysis_summary.json').write_text(json.dumps(summary_out,indent=2),encoding='utf-8')
    print(json.dumps(summary_out,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory outside frozen_analysis_v3; existing paths are refused.')
    parser.add_argument('--tables-only', action='store_true', help='Skip figure exports.')
    args = parser.parse_args()
    main(args.output, args.tables_only)
