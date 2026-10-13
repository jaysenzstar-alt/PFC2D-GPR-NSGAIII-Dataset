# Direct calibration checks

The twelve cases use the full-precision vectors in `model_files/representative_candidates_nsgaIII.csv`. Each case independently restores `model_files/yuya.p2sav`, assigns the five parameters through the supplied bonding script and follows the same uniaxial loading procedure. The specimen contains 6635 particles. Drivers request 24 processors.

## Extract the supplied histories

With Python and NumPy installed, run from the archive root:

```text
python direct_validation/validation.py analyze --output reproduced_direct
```

This reads `recorded_histories/current_01/history.his` through `current_12/history.his`. The new output directory contains `joint_responses.json`, including all sensitivity readouts, and `target_gpr_pfc_comparison.csv`. It must not already exist. This command performs numerical extraction only, not a new PFC simulation.

The history columns are cycle, stress (MPa), wall axial strain, Poisson's ratio, secant modulus (GPa), integrated local axial strain, integrated local lateral strain and crack count. Peak stress and peak strain come from the same maximum-stress record. For modulus and Poisson's ratio, the script selects the ascending-branch record nearest 50% of peak stress, taking the first record in a tie. Sensitivity readouts use 25%, 40% and 60%. Modulus is a secant quantity, not a regression slope. Poisson's ratio is the absolute ratio of integrated lateral to axial local strain.

Errors are absolute differences divided by the absolute target. MARE is the unweighted mean of the four errors. The PFC–GPR gap also uses the target as denominator. The sampling positions used for all historical training and target responses are not established by this extraction procedure.

## Prepare and run native simulations

Use PFC2D 5.00 Release 025 and a compatible `fishuser_function005_64.dll` from your licensed installation. These runtime components are not distributed here. To prepare separate working directories, supply the plug-in path:

```text
python direct_validation/validation.py prepare-runs --output reproduced_runs --plugin "PATH_TO_PLUGIN/fishuser_function005_64.dll"
```

This command copies the specimen, native support scripts and exact driver into twelve new directories; it does not launch PFC. In each directory, start the PFC2D console with that directory as its working directory and enter:

```text
call launch.p2dat
```

The driver loads the plug-in before restoring the specimen, applies the parameters, resets the integrated strains and peak-stress accumulator, and runs loading until the stress falls below 10% of its peak. It writes `history.his`, a log and `recovered.p2sav`, then exits. Keep a failed or incomplete attempt separate; do not treat it as a valid response. After all twelve runs finish, extract them with:

```text
python direct_validation/validation.py analyze --histories reproduced_runs --output reproduced_run_responses
```

The historical specimen-generation and preloading scripts in `generation_sources` document the construction procedure. They are not needed when restoring the supplied specimen and should not be substituted for it when reproducing these twelve checks.

## Reanalyse the archived numerical results

Install the Python dependencies from the archive-root `requirements.txt`. From the archive root, use a new output directory:

```text
python surrogate_analysis/analysis_code/reanalyze_round3.py --output reproduced/round3 --tables-only
python surrogate_analysis/modeling_code/reproduce_grey_relational_analysis.py --dataset surrogate_analysis/input_data/3125_full_factorial_dataset.csv --reference surrogate_analysis/modeling_results/grey_relational/fig05_grey_relational_matrix_reproduced.csv --output-dir reproduced/grey
python final_tables/recompute_historical_errors.py reproduced/historical_errors.csv
```

Omit `--tables-only` to export the figures as well. The round-3 script aggregates the supplied run metrics and final populations; it does not fit models or run the optimizer. Its fixed counts refer to this archive, not arbitrary newly generated populations. Historical errors recomputed from rounded displayed responses remain separate from the current full-precision direct checks.

## Use a newly fitted MATLAB model

The default MATLAB entry points read the supplied frozen model and run records. They do not automatically use a model from a previous refitting command. To propagate a newly fitted model, start MATLAB in the archive root and explicitly connect the paths:

```matlab
addpath(fullfile(pwd, 'surrogate_analysis', 'modeling_code'));
run_round2_surrogate_refit('review');
modelFile = fullfile(pwd, 'surrogate_analysis', 'modeling_results', ...
    'reproduced', 'surrogate_refit_review', 'round2_production_models.mat');
optimizerDir = fullfile(pwd, 'reproduced', 'optimizer_new_model');
run_round2_nsga_comparison('review', modelFile, optimizerDir);
runFile = fullfile(optimizerDir, 'nsga_comparison_runs.csv');
export_round2_representative_candidates(modelFile, runFile, ...
    fullfile(pwd, 'reproduced', 'representatives_new_model.csv'));
run_round2_forward_check(modelFile, ...
    fullfile(pwd, 'reproduced', 'forward_check_new_model'));
```

These commands require MATLAB and the Statistics and Machine Learning Toolbox. Use new output locations; the refitting script refuses a nonempty output directory. The `review` optimizer mode uses the reported 60-individual, 30-generation budget with ten runs per target. Newly fitted models and newly selected representatives are separate results: the twelve supplied PFC histories validate only the frozen representative vectors, not new vectors produced by this sequence.
