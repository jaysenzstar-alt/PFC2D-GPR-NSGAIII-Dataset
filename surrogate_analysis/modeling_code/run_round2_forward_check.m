function run_round2_forward_check(modelFile, outputDir)
%RUN_ROUND2_FORWARD_CHECK Evaluate the refitted production surrogates on
%the seven retained forward-check cases that lie inside all Table 1 bounds.
% With no arguments, use archived models. Optional arguments specify a model
% file and a separate output directory for a new model version.

codeDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(codeDir);
inputDir = fullfile(rootDir, 'input_data');
modelDir = fullfile(rootDir, 'trained_models');
resultsDir = fullfile(rootDir, 'modeling_results');

if nargin < 1 || isempty(modelFile)
    modelFile = fullfile(modelDir, 'round2_production_models.mat');
end
inputFile = fullfile(inputDir, 'forward_check_7_in_domain.csv');
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(resultsDir, 'reproduced', 'forward_check');
end
ensure_empty_output_dir(outputDir);

raw = load(modelFile, 'modelBundle');
if ~isfield(raw, 'modelBundle')
    error('Expected modelBundle in %s.', modelFile);
end
models = raw.modelBundle;
T = readtable(inputFile);
if height(T) ~= 7 || ~isequal(T.sample(:)', [1:6, 8])
    error('Expected seven forward-check cases numbered 1-6 and 8.');
end

X = [T.pb_emod_gpa, T.pb_kra, T.pb_ten_mpa, T.pb_coh_mpa, T.pb_fa_deg];
if any(X < models.varMin, 'all') || any(X > models.varMax, 'all')
    error('Forward-check input contains a parameter outside the reviewed domain.');
end
Y = [T.actual_peak_stress_mpa, T.actual_peak_strain_permille, ...
    T.actual_elastic_modulus_gpa, T.actual_poisson_ratio];

Yhat = zeros(height(T), 4);
for i = 1:height(T)
    Yhat(i, :) = surrogate_predict(X(i, :), models);
end
rel = abs(Yhat-Y)./max(abs(Y), eps);

Predictions = [T, array2table(Yhat, 'VariableNames', ...
    {'pred_peak_stress_mpa','pred_peak_strain_permille', ...
    'pred_elastic_modulus_gpa','pred_poisson_ratio'}), ...
    array2table(rel, 'VariableNames', ...
    {'relerr_peak_stress','relerr_peak_strain', ...
    'relerr_elastic_modulus','relerr_poisson_ratio'})];
writetable(Predictions, fullfile(outputDir, 'forward_check_predictions.csv'));

response = string({'peak_stress_mpa','peak_strain_permille', ...
    'elastic_modulus_gpa','poisson_ratio'})';
mse = mean((Yhat-Y).^2, 1)';
rmse = sqrt(mse);
mare = mean(rel, 1)';
r2 = zeros(4, 1);
for j = 1:4
    r2(j) = 1-sum((Yhat(:,j)-Y(:,j)).^2)/sum((Y(:,j)-mean(Y(:,j))).^2);
end
Metrics = table(response, repmat(height(T),4,1), mse, rmse, mare, r2, ...
    'VariableNames', {'response','n_cases','mse','rmse','mare','r2'});
writetable(Metrics, fullfile(outputDir, 'forward_check_metrics.csv'));

manifest = fopen(fullfile(outputDir, 'RUN_MANIFEST.txt'), 'wt');
cleanup = onCleanup(@() fclose(manifest));
fprintf(manifest, 'script=run_round2_forward_check\n');
fprintf(manifest, 'matlab_version=%s\n', version);
fprintf(manifest, 'model_file=%s\n', modelFile);
fprintf(manifest, 'input_file=%s\n', inputFile);
fprintf(manifest, 'n_cases=7\n');
fprintf(manifest, 'scope=retained cases inside all five Table 1 parameter bounds\n');
fprintf(manifest, 'excluded_submitted_cases=7,9-16 because one or more inputs lie outside the reviewed training domain\n');

fprintf('Round-2 in-domain forward check complete: %s\n', outputDir);
end


function y = surrogate_predict(position, models)
x = position(:)';
xn = (x-models.xScale.offset)./models.xScale.range;
y = zeros(1, 4);
for j = 1:4
    yn = predict(models.models{j}, xn);
    y(j) = yn.*models.yScales{j}.range+models.yScales{j}.offset;
end
end


function ensure_empty_output_dir(outDir)
if isfolder(outDir)
    existing = dir(outDir);
    existing = existing(~ismember({existing.name}, {'.','..'}));
    if ~isempty(existing)
        error('Refusing to overwrite nonempty output directory: %s', outDir);
    end
else
    mkdir(outDir);
end
end
