function run_round2_surrogate_refit(mode)
%RUN_ROUND2_SURROGATE_REFIT Scale-invariant GPR selection and final refit.
%
% Kernel selection and fitting use the following data separation:
%   1. five deterministic 70/10/20 train/validation/test splits;
%   2. min-max scaling fitted on the training subset only during screening;
%   3. kernel selection separately for each response by validation RMSE
%      divided by the corresponding training-response range;
%   4. refit on train+validation and one evaluation on the untouched test set;
%   5. production-kernel selection by mean validation NRMSE across repeats;
%   6. final production models fitted on all 3,125 rows.
%
% Output: modeling_results/reproduced/surrogate_refit_<mode>/.
% Pass the resulting model file explicitly to run_round2_nsga_comparison.
% The output directory must be absent or empty.

if nargin < 1 || isempty(mode)
    mode = "review";
end
mode = lower(string(mode));
cfg = local_config(mode);

codeDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(codeDir);
inputDir = fullfile(rootDir, 'input_data');
resultsDir = fullfile(rootDir, 'modeling_results');

dataFile = fullfile(inputDir, '3125.xlsx');
if ~isfile(dataFile)
    error('Missing reviewed training dataset: %s', dataFile);
end

outDir = fullfile(resultsDir, 'reproduced', "surrogate_refit_" + mode);
ensure_empty_output_dir(outDir);

T = readtable(dataFile, 'VariableNamingRule', 'preserve');
X = table2array(T(:, 1:5));
Y = table2array(T(:, 6:9));
if size(X, 1) ~= 3125 || size(X, 2) ~= 5 || size(Y, 2) ~= 4
    error('Expected 3,125 rows, five predictors, and four responses.');
end
if any(~isfinite(X), 'all') || any(~isfinite(Y), 'all')
    error('Training data contain nonfinite values.');
end
if size(unique(X, 'rows'), 1) ~= size(X, 1)
    error('The five-parameter factorial grid contains duplicate rows.');
end

responseNames = {'peak_stress_mpa', 'peak_strain_permille', ...
    'elastic_modulus_gpa', 'poisson_ratio'};
predictorNames = {'bond_effective_modulus_gpa', 'stiffness_ratio', ...
    'tensile_strength_mpa', 'cohesion_mpa', 'friction_angle_deg'};

n = size(X, 1);
nTest = round(cfg.testFraction*n);
nValidation = round(cfg.validationFraction*n);
kernelRows = cell(cfg.nRepeats*numel(responseNames)*numel(cfg.kernels), 10);
nestedRows = cell(cfg.nRepeats*numel(responseNames), 16);
kernelRow = 0;
nestedRow = 0;

for r = 1:cfg.nRepeats
    splitSeed = cfg.splitSeedBase + r;
    rng(splitSeed, 'twister');
    order = randperm(n);
    idxTest = order(1:nTest);
    idxValidation = order(nTest+1:nTest+nValidation);
    idxTrain = order(nTest+nValidation+1:end);

    [XTrainN, xTrainScale] = minmax_fit(X(idxTrain, :));
    XValidationN = minmax_apply(X(idxValidation, :), xTrainScale);
    idxFit = [idxTrain idxValidation];
    [XFitN, xFitScale] = minmax_fit(X(idxFit, :));
    XTestN = minmax_apply(X(idxTest, :), xFitScale);

    for j = 1:numel(responseNames)
        [yTrainN, yTrainScale] = minmax_fit(Y(idxTrain, j));
        validationNrmse = nan(numel(cfg.kernels), 1);
        validationRmse = nan(numel(cfg.kernels), 1);

        for k = 1:numel(cfg.kernels)
            fitSeed = splitSeed + 10000*k + 100*j;
            rng(fitSeed, 'twister');
            mdl = fit_one_gpr(XTrainN, yTrainN, cfg.kernels{k}, cfg);
            predValidation = minmax_reverse(predict(mdl, XValidationN), yTrainScale);
            errValidation = predValidation - Y(idxValidation, j);
            validationRmse(k) = sqrt(mean(errValidation.^2));
            responseRange = max(Y(idxTrain, j)) - min(Y(idxTrain, j));
            validationNrmse(k) = validationRmse(k)/max(responseRange, eps);

            kernelRow = kernelRow + 1;
            kernelRows(kernelRow, :) = {r, splitSeed, responseNames{j}, ...
                cfg.kernels{k}, fitSeed, numel(idxTrain), numel(idxValidation), ...
                validationRmse(k), validationNrmse(k), responseRange};
        end

        [~, bestIndex] = min(validationNrmse);
        selectedKernel = cfg.kernels{bestIndex};
        finalFitSeed = splitSeed + 50000 + 100*j;
        rng(finalFitSeed, 'twister');
        [yFitN, yFitScale] = minmax_fit(Y(idxFit, j));
        mdl = fit_one_gpr(XFitN, yFitN, selectedKernel, cfg);
        predTest = minmax_reverse(predict(mdl, XTestN), yFitScale);
        metrics = regression_metrics(Y(idxTest, j), predTest);

        nestedRow = nestedRow + 1;
        nestedRows(nestedRow, :) = {r, splitSeed, responseNames{j}, ...
            selectedKernel, finalFitSeed, numel(idxTrain), numel(idxValidation), ...
            numel(idxTest), metrics.rmse, metrics.nrmse, metrics.mae, ...
            metrics.mare, metrics.r2, min(Y(idxFit, j)), max(Y(idxFit, j)), ...
            max(Y(idxFit, j))-min(Y(idxFit, j))};
    end
end

KernelRuns = cell2table(kernelRows, 'VariableNames', ...
    {'repeat','split_seed','response','kernel','fit_seed','n_train', ...
    'n_validation','validation_rmse','validation_nrmse','training_response_range'});
NestedTest = cell2table(nestedRows, 'VariableNames', ...
    {'repeat','split_seed','response','selected_kernel','fit_seed','n_train', ...
    'n_validation','n_test','test_rmse','test_nrmse','test_mae','test_mare', ...
    'test_r2','fit_response_min','fit_response_max','fit_response_range'});

writetable(KernelRuns, fullfile(outDir, 'kernel_validation_runs.csv'));
writetable(NestedTest, fullfile(outDir, 'nested_selected_kernel_test_runs.csv'));

[ProductionSelection, KernelSummary] = select_production_kernels( ...
    KernelRuns, responseNames, cfg.kernels);
writetable(KernelSummary, fullfile(outDir, 'kernel_validation_summary.csv'));
writetable(ProductionSelection, fullfile(outDir, 'production_kernel_selection.csv'));

NestedSummary = groupsummary(NestedTest, 'response', {'mean','std'}, ...
    {'test_rmse','test_nrmse','test_mae','test_mare','test_r2'});
writetable(NestedSummary, fullfile(outDir, 'nested_selected_kernel_test_summary.csv'));

[XAllN, xAllScale] = minmax_fit(X);
productionModels = cell(1, numel(responseNames));
yScales = cell(1, numel(responseNames));
trainingRows = cell(numel(responseNames), 10);

for j = 1:numel(responseNames)
    kernel = ProductionSelection.selected_kernel{j};
    productionSeed = cfg.productionSeedBase + j;
    rng(productionSeed, 'twister');
    [yAllN, yScales{j}] = minmax_fit(Y(:, j));
    productionModels{j} = fit_one_gpr(XAllN, yAllN, kernel, cfg);
    predAll = minmax_reverse(predict(productionModels{j}, XAllN), yScales{j});
    metrics = regression_metrics(Y(:, j), predAll);
    trainingRows(j, :) = {responseNames{j}, kernel, productionSeed, n, ...
        metrics.rmse, metrics.nrmse, metrics.mae, metrics.mare, metrics.r2, ...
        max(Y(:, j))-min(Y(:, j))};
end

TrainingFit = cell2table(trainingRows, 'VariableNames', ...
    {'response','selected_kernel','fit_seed','n','rmse','nrmse','mae','mare', ...
    'r2','response_range'});
writetable(TrainingFit, fullfile(outDir, 'production_training_fit_metrics.csv'));

modelBundle.schemaVersion = 1;
modelBundle.createdBy = mfilename;
modelBundle.matlabVersion = version;
modelBundle.datasetRelativePath = fullfile('input_data', '3125.xlsx');
modelBundle.predictorNames = predictorNames;
modelBundle.responseNames = responseNames;
modelBundle.varMin = min(X, [], 1);
modelBundle.varMax = max(X, [], 1);
modelBundle.xScale = xAllScale;
modelBundle.yScales = yScales;
modelBundle.models = productionModels;
modelBundle.productionSelection = ProductionSelection;
modelBundle.fitOptions = cfg.fitOptions;
modelBundle.splitSeedBase = cfg.splitSeedBase;
modelBundle.productionSeedBase = cfg.productionSeedBase;
modelBundle.selectionCriterion = [ ...
    'lowest mean validation RMSE divided by the training-response range, ' ...
    'selected separately for each response across deterministic repeats'];
save(fullfile(outDir, 'round2_production_models.mat'), 'modelBundle', '-v7.3');

write_manifest(outDir, cfg, n, nTest, nValidation, ProductionSelection);
fprintf('Round-2 surrogate refit complete: %s\n', outDir);
end

function mdl = fit_one_gpr(X, y, kernel, cfg)
mdl = fitrgp(X, y, ...
    'BasisFunction', cfg.fitOptions.BasisFunction, ...
    'KernelFunction', kernel, ...
    'FitMethod', cfg.fitOptions.FitMethod, ...
    'PredictMethod', cfg.fitOptions.PredictMethod, ...
    'ActiveSetSize', min(cfg.fitOptions.ActiveSetSize, size(X, 1)));
end

function [Selection, Summary] = select_production_kernels(KernelRuns, responses, kernels)
summaryRows = cell(numel(responses)*numel(kernels), 7);
selectionRows = cell(numel(responses), 5);
summaryRow = 0;
for j = 1:numel(responses)
    means = nan(numel(kernels), 1);
    standardDeviations = nan(numel(kernels), 1);
    for k = 1:numel(kernels)
        mask = strcmp(KernelRuns.response, responses{j}) & ...
            strcmp(KernelRuns.kernel, kernels{k});
        values = KernelRuns.validation_nrmse(mask);
        means(k) = mean(values);
        standardDeviations(k) = std(values);
        summaryRow = summaryRow + 1;
        summaryRows(summaryRow, :) = {responses{j}, kernels{k}, numel(values), ...
            mean(values), std(values), min(values), max(values)};
    end
    [bestMean, bestIndex] = min(means);
    selectionRows(j, :) = {responses{j}, kernels{bestIndex}, bestMean, ...
        standardDeviations(bestIndex), numel(kernels)};
end
Summary = cell2table(summaryRows, 'VariableNames', ...
    {'response','kernel','n_repeats','mean_validation_nrmse', ...
    'std_validation_nrmse','min_validation_nrmse','max_validation_nrmse'});
Selection = cell2table(selectionRows, 'VariableNames', ...
    {'response','selected_kernel','mean_validation_nrmse', ...
    'std_validation_nrmse','candidate_kernel_count'});
end

function metrics = regression_metrics(y, yhat)
err = yhat-y;
metrics.rmse = sqrt(mean(err.^2));
responseRange = max(y)-min(y);
metrics.nrmse = metrics.rmse/max(responseRange, eps);
metrics.mae = mean(abs(err));
metrics.mare = mean(abs(err)./max(abs(y), eps));
ssRes = sum(err.^2);
ssTot = sum((y-mean(y)).^2);
metrics.r2 = 1-ssRes/max(ssTot, eps);
end

function [scaled, params] = minmax_fit(values)
params.offset = min(values, [], 1);
params.range = max(values, [], 1)-params.offset;
params.range(params.range == 0) = 1;
scaled = (values-params.offset)./params.range;
end

function scaled = minmax_apply(values, params)
scaled = (values-params.offset)./params.range;
end

function values = minmax_reverse(scaled, params)
values = scaled.*params.range+params.offset;
end

function ensure_empty_output_dir(outDir)
if isfolder(outDir)
    listing = dir(outDir);
    listing = listing(~ismember({listing.name}, {'.', '..'}));
    if ~isempty(listing)
        error('Refusing to overwrite nonempty output directory: %s', outDir);
    end
else
    mkdir(outDir);
end
end

function write_manifest(outDir, cfg, n, nTest, nValidation, Selection)
manifestPath = fullfile(outDir, 'RUN_MANIFEST.txt');
fid = fopen(manifestPath, 'wt');
if fid < 0
    error('Could not create manifest: %s', manifestPath);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'script=%s\n', mfilename);
fprintf(fid, 'matlab_version=%s\n', version);
fprintf(fid, 'n_rows=%d\n', n);
fprintf(fid, 'n_repeats=%d\n', cfg.nRepeats);
fprintf(fid, 'split_seed_base=%d\n', cfg.splitSeedBase);
fprintf(fid, 'n_train=%d\n', n-nTest-nValidation);
fprintf(fid, 'n_validation=%d\n', nValidation);
fprintf(fid, 'n_test=%d\n', nTest);
fprintf(fid, 'validation_criterion=RMSE divided by training-response range\n');
for j = 1:height(Selection)
    fprintf(fid, 'selected_%s=%s\n', Selection.response{j}, ...
        Selection.selected_kernel{j});
end
end

function cfg = local_config(mode)
cfg.splitSeedBase = 20260710;
cfg.productionSeedBase = 202608100;
cfg.validationFraction = 0.10;
cfg.testFraction = 0.20;
cfg.kernels = {'exponential','ardexponential','squaredexponential', ...
    'ardsquaredexponential','ardrationalquadratic','matern32', ...
    'ardmatern32','matern52','ardmatern52','rationalquadratic'};
cfg.fitOptions.BasisFunction = 'constant';
cfg.fitOptions.FitMethod = 'sr';
cfg.fitOptions.PredictMethod = 'fic';
cfg.fitOptions.ActiveSetSize = 1000;
switch mode
    case "smoke"
        cfg.nRepeats = 1;
        cfg.kernels = {'ardmatern32','ardmatern52','ardrationalquadratic'};
    case "review"
        cfg.nRepeats = 5;
    case "full"
        cfg.nRepeats = 20;
    otherwise
        error('Unknown mode: %s. Use smoke, review, or full.', mode);
end
end
