function export_round2_representative_candidates(modelFile, runFile, outputFile)
%EXPORT_ROUND2_REPRESENTATIVE_CANDIDATES Select and verify one NSGA-III
%candidate per numerical target from ten NSGA-III runs.
% Defaults use the archived model and run table. To check newly generated
% results, pass the corresponding modelFile, runFile and outputFile together.

codeDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(codeDir);
inputDir = fullfile(rootDir, 'input_data');
modelDir = fullfile(rootDir, 'trained_models');
resultsDir = fullfile(rootDir, 'modeling_results');
outputDir = fullfile(resultsDir, 'reproduced');

if nargin < 1 || isempty(modelFile)
    modelFile = fullfile(modelDir, 'round2_production_models.mat');
end
if nargin < 2 || isempty(runFile)
    runFile = fullfile(resultsDir, 'optimizer', 'nsga_comparison_runs.csv');
end
if nargin < 3 || isempty(outputFile)
    outputFile = fullfile(outputDir, 'representative_candidates_nsgaIII.csv');
else
    outputDir = fileparts(char(outputFile));
    if isempty(outputDir), outputDir = pwd; end
end
targetsFile = fullfile(inputDir, 'targets_12.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end
if isfile(outputFile)
    error('Refusing to overwrite existing representative-candidate file: %s', outputFile);
end
if ~isfile(modelFile) || ~isfile(runFile) || ~isfile(targetsFile)
    error('Required model, target, or run-level result file is missing.');
end

raw = load(modelFile, 'modelBundle');
if ~isfield(raw, 'modelBundle')
    error('Expected modelBundle in %s.', modelFile);
end
models = raw.modelBundle;
runs = readtable(runFile);
runs = runs(strcmp(string(runs.algorithm), "NSGAIII"), :);
targets = readtable(targetsFile);

samples = unique(runs.sample, 'sorted');
if ~isequal(samples(:)', 1:12)
    error('Expected NSGA-III results for samples 1-12.');
end

n = numel(samples);
selectedRun = zeros(n, 1);
selectedSeed = zeros(n, 1);
target = zeros(n, 4);
position = zeros(n, 5);
prediction = zeros(n, 4);
errors = zeros(n, 4);
meanError = zeros(n, 1);

for i = 1:n
    sampleRows = runs(runs.sample == samples(i), :);
    if height(sampleRows) ~= 10
        error('Expected 10 NSGA-III runs for sample %d.', samples(i));
    end
    [~, localIndex] = min(sampleRows.best_mean_relative_error);
    row = sampleRows(localIndex, :);

    selectedRun(i) = row.run;
    selectedSeed(i) = row.seed;
    targetRow = targets(targets.sample == samples(i), :);
    if height(targetRow) ~= 1
        error('Expected one target row for sample %d.', samples(i));
    end
    target(i, :) = [targetRow.target_peak_stress_mpa, ...
        targetRow.target_peak_strain_permille, targetRow.target_elastic_modulus_gpa, ...
        targetRow.target_poisson_ratio];
    position(i, :) = [row.best_pb_emod_gpa, row.best_pb_kra, ...
        row.best_pb_ten_mpa, row.best_pb_coh_mpa, row.best_pb_fa_deg];
    prediction(i, :) = surrogate_predict(position(i, :), models);
    errors(i, :) = abs(prediction(i, :) - target(i, :))./max(abs(target(i, :)), eps);
    meanError(i) = mean(errors(i, :));

    storedErrors = [row.best_err_peak_stress, row.best_err_peak_strain, ...
        row.best_err_elastic_modulus, row.best_err_poisson_ratio];
    if max(abs(errors(i, :) - storedErrors)) > 1e-10 || ...
            abs(meanError(i) - row.best_mean_relative_error) > 1e-10
        error('Independent prediction does not match stored run metrics for sample %d.', samples(i));
    end
end

T = table(samples, selectedRun, selectedSeed, ...
    target(:,1), target(:,2), target(:,3), target(:,4), ...
    position(:,1), position(:,2), position(:,3), position(:,4), position(:,5), ...
    prediction(:,1), prediction(:,2), prediction(:,3), prediction(:,4), ...
    errors(:,1), errors(:,2), errors(:,3), errors(:,4), meanError, ...
    'VariableNames', {'sample','selected_run','seed', ...
    'target_peak_stress_mpa','target_peak_strain_permille', ...
    'target_elastic_modulus_gpa','target_poisson_ratio', ...
    'pb_emod_gpa','pb_kra','pb_ten_mpa','pb_coh_mpa','pb_fa_deg', ...
    'pred_peak_stress_mpa','pred_peak_strain_permille', ...
    'pred_elastic_modulus_gpa','pred_poisson_ratio', ...
    'err_peak_stress','err_peak_strain','err_elastic_modulus', ...
    'err_poisson_ratio','mean_relative_error'});

writetable(T, outputFile);
fprintf('Representative candidates written: %s\n', outputFile);
end


function y = surrogate_predict(position, models)
x = position(:)';
if any(x < models.varMin) || any(x > models.varMax)
    error('Surrogate prediction requested outside the reviewed training domain.');
end
xn = (x-models.xScale.offset)./models.xScale.range;
y = zeros(1, 4);
for j = 1:4
    yn = predict(models.models{j}, xn);
    y(j) = yn.*models.yScales{j}.range+models.yScales{j}.offset;
end
end
