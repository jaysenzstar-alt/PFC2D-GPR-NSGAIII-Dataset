function run_surrogate_model_audit(mode)
%RUN_SURROGATE_MODEL_AUDIT Repeated disjoint train/validation/test audit.
%
% The audit uses a 70/10/20 split by unique row index. Kernel selection is
% performed only on the validation subset, and the final metrics are then
% evaluated once on the untouched test subset. Scaling is fitted on training
% during kernel screening and on training plus validation for the final refit.
% Every repeat exports its seed and split sizes. Checkpoints permit restart.

if nargin < 1 || isempty(mode)
    mode = 'review';
end
mode = lower(string(mode));
cfg = local_config(mode);

thisDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(thisDir);
% Use the supplied training dataset.
dataFile = fullfile(rootDir, '..', 'input_data', '3125.xlsx');
if ~isfile(dataFile)
    error('Packaged input is missing: %s', dataFile);
end

T = readtable(dataFile, 'VariableNamingRule', 'preserve');
X = table2array(T(:, 1:5));
Y = table2array(T(:, 6:9));
responses = {'peak_stress_mpa', 'peak_strain_permille', ...
    'elastic_modulus_gpa', 'poisson_ratio'};

if size(unique(X, 'rows'), 1) ~= size(X, 1)
    error('The five-parameter grid contains duplicated rows.');
end

outDir = fullfile(rootDir, "reproduced", "results_surrogate_" + mode);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

checkpointFile = fullfile(outDir, 'surrogate_resume_checkpoint.mat');
runsFile = fullfile(outDir, 'surrogate_repeated_runs.csv');
runNN = cfg.runNeuralNetwork && exist('fitrnet', 'file') == 2;
expectedRows = cfg.nRepeats*numel(responses)*(1 + double(runNN));
if exist(runsFile, 'file')
    existingRuns = readtable(runsFile, 'VariableNamingRule', 'preserve');
    if height(existingRuns) == expectedRows ...
            && ismember('fit_seed', existingRuns.Properties.VariableNames)
        if exist(checkpointFile, 'file')
            delete(checkpointFile);
        end
        fprintf('Surrogate audit already complete: %s\n', outDir);
        return;
    end
end

metricRecords = cell(0, 13);
kernelRecords = cell(0, 9);
completedRepeats = 0;
if exist(checkpointFile, 'file')
    saved = load(checkpointFile, 'state');
    state = saved.state;
    if ~isfield(state, 'schemaVersion') || state.schemaVersion ~= 2 ...
            || ~strcmp(state.mode, char(mode)) || state.nRepeats ~= cfg.nRepeats
        error('Incompatible checkpoint: %s', checkpointFile);
    end
    metricRecords = state.metricRecords;
    kernelRecords = state.kernelRecords;
    completedRepeats = state.completedRepeats;
    fprintf('Resuming surrogate audit after repeat %d/%d.\n', ...
        completedRepeats, cfg.nRepeats);
end
metricNo = size(metricRecords, 1);
kernelNo = size(kernelRecords, 1);
n = size(X, 1);
nTest = round(cfg.testFraction*n);
nValidation = round(cfg.validationFraction*n);

for r = completedRepeats+1:cfg.nRepeats
    seed = cfg.seedBase + r;
    rng(seed, 'twister');
    order = randperm(n);
    idxTest = order(1:nTest);
    idxValidation = order(nTest+1:nTest+nValidation);
    idxTrain = order(nTest+nValidation+1:end);

    [XTrainN, xScale] = minmax_fit(X(idxTrain, :));
    XValidationN = minmax_apply(X(idxValidation, :), xScale);

    kernelScores = nan(numel(cfg.kernels), 1);
    responseScores = nan(numel(cfg.kernels), numel(responses));
    for iRecord = 1:size(kernelRecords, 1)
        if kernelRecords{iRecord, 1} == r
            existingKernel = find(strcmp(cfg.kernels, kernelRecords{iRecord, 3}), 1);
            if ~isempty(existingKernel)
                kernelScores(existingKernel) = kernelRecords{iRecord, 5};
                responseScores(existingKernel, :) = cell2mat(kernelRecords(iRecord, 6:9));
            end
        end
    end
    for k = 1:numel(cfg.kernels)
        if isfinite(kernelScores(k))
            continue;
        end
        kernelFitSeedBase = seed + 10000*k;
        for j = 1:numel(responses)
            rng(kernelFitSeedBase + j, 'twister');
            [yTrainN, yScale] = minmax_fit(Y(idxTrain, j));
            mdl = fitrgp(XTrainN, yTrainN, ...
                'BasisFunction', 'constant', ...
                'KernelFunction', cfg.kernels{k}, ...
                'FitMethod', 'sr', ...
                'PredictMethod', 'fic', ...
                'ActiveSetSize', min(1000, size(XTrainN, 1)));
            predN = predict(mdl, XValidationN);
            pred = minmax_reverse(predN, yScale);
            responseRange = max(Y(idxTrain, j)) - min(Y(idxTrain, j));
            responseScores(k, j) = sqrt(mean((Y(idxValidation, j)-pred).^2)) ...
                / max(responseRange, eps);
        end
        kernelScores(k) = mean(responseScores(k, :));
        kernelNo = kernelNo + 1;
        kernelRecords(kernelNo, :) = {r, seed, cfg.kernels{k}, kernelFitSeedBase, ...
            kernelScores(k), responseScores(k, 1), responseScores(k, 2), ...
            responseScores(k, 3), responseScores(k, 4)}; %#ok<AGROW>
        state = checkpoint_state(mode, cfg.nRepeats, completedRepeats, ...
            metricRecords, kernelRecords);
        save_checkpoint(checkpointFile, state);
        fprintf('Repeat %d/%d kernel %d/%d complete: %s\n', ...
            r, cfg.nRepeats, k, numel(cfg.kernels), cfg.kernels{k});
    end

    [~, bestKernelIndex] = min(kernelScores);
    selectedKernel = cfg.kernels{bestKernelIndex};
    idxFit = [idxTrain idxValidation];
    [XFitN, xScale] = minmax_fit(X(idxFit, :));
    XTestN = minmax_apply(X(idxTest, :), xScale);

    for j = 1:numel(responses)
        gprFitSeed = seed + 20000 + j;
        rng(gprFitSeed, 'twister');
        [yFitN, yScale] = minmax_fit(Y(idxFit, j));
        mdl = fitrgp(XFitN, yFitN, ...
            'BasisFunction', 'constant', ...
            'KernelFunction', selectedKernel, ...
            'FitMethod', 'sr', ...
            'PredictMethod', 'fic', ...
            'ActiveSetSize', min(1000, size(XFitN, 1)));
        predN = predict(mdl, XTestN);
        pred = minmax_reverse(predN, yScale);
        metricNo = metricNo + 1;
        metricRecords(metricNo, :) = metric_row(r, seed, responses{j}, ...
            "GPR", selectedKernel, gprFitSeed, numel(idxTrain), numel(idxValidation), ...
            numel(idxTest), Y(idxTest, j), pred); %#ok<AGROW>

        if runNN
            nnFitSeed = seed + 30000 + j;
            rng(nnFitSeed, 'twister');
            net = fitrnet(X(idxFit, :), Y(idxFit, j), ...
                'LayerSizes', cfg.netLayerSizes, ...
                'Standardize', true, ...
                'IterationLimit', cfg.netIterationLimit);
            pred = predict(net, X(idxTest, :));
            metricNo = metricNo + 1;
            metricRecords(metricNo, :) = metric_row(r, seed, responses{j}, ...
                "NN", "not_applicable", nnFitSeed, numel(idxTrain), numel(idxValidation), ...
                numel(idxTest), Y(idxTest, j), pred); %#ok<AGROW>
        end
    end

    completedRepeats = r;
    state = checkpoint_state(mode, cfg.nRepeats, completedRepeats, ...
        metricRecords, kernelRecords);
    save_checkpoint(checkpointFile, state);
    write_outputs(outDir, metricRecords, kernelRecords);
    fprintf('Repeat %d/%d complete: seed=%d, selected kernel=%s\n', ...
        r, cfg.nRepeats, seed, selectedKernel);
end

write_outputs(outDir, metricRecords, kernelRecords);
if exist(checkpointFile, 'file')
    delete(checkpointFile);
end

fprintf('Surrogate audit written to %s\n', outDir);
end

function state = checkpoint_state(mode, nRepeats, completedRepeats, ...
    metricRecords, kernelRecords)
state.schemaVersion = 2;
state.mode = char(mode);
state.nRepeats = nRepeats;
state.completedRepeats = completedRepeats;
state.metricRecords = metricRecords;
state.kernelRecords = kernelRecords;
end

function save_checkpoint(checkpointFile, state)
checkpointFile = char(checkpointFile);
tempFile = [checkpointFile '.tmp.mat'];
save(tempFile, 'state', '-v7');
movefile(tempFile, checkpointFile, 'f');
end

function write_outputs(outDir, metricRecords, kernelRecords)
Runs = cell2table(metricRecords, 'VariableNames', ...
    {'repeat','seed','response','model','selected_kernel','fit_seed','n_train', ...
    'n_validation','n_test','rmse','mae','mare','r2'});
writetable(Runs, fullfile(outDir, 'surrogate_repeated_runs.csv'));

Summary = groupsummary(Runs, {'response','model'}, {'mean','std'}, ...
    {'rmse','mae','mare','r2'});
writetable(Summary, fullfile(outDir, 'surrogate_repeated_summary.csv'));

KernelAudit = cell2table(kernelRecords, 'VariableNames', ...
    {'repeat','seed','kernel','fit_seed_base','mean_validation_nrmse', ...
    'peak_stress_validation_nrmse','peak_strain_validation_nrmse', ...
    'elastic_modulus_validation_nrmse','poisson_ratio_validation_nrmse'});
writetable(KernelAudit, fullfile(outDir, 'kernel_selection_runs.csv'));

Selected = Runs(Runs.model == "GPR", {'repeat','seed','selected_kernel', ...
    'n_train','n_validation','n_test'});
Selected = unique(Selected, 'rows');
writetable(Selected, fullfile(outDir, 'selected_kernel_by_repeat.csv'));
end

function cfg = local_config(mode)
cfg.seedBase = 20260710;
cfg.validationFraction = 0.10;
cfg.testFraction = 0.20;
cfg.netLayerSizes = 10;
cfg.runNeuralNetwork = true;
cfg.kernels = {'exponential','ardexponential','squaredexponential', ...
    'ardsquaredexponential','ardrationalquadratic','matern32', ...
    'ardmatern32','matern52','ardmatern52','rationalquadratic'};
switch mode
    case "smoke"
        cfg.nRepeats = 1;
        cfg.netIterationLimit = 50;
        cfg.kernels = {'ardsquaredexponential','ardrationalquadratic'};
    case "quick"
        cfg.nRepeats = 3;
        cfg.netIterationLimit = 150;
    case "review"
        cfg.nRepeats = 5;
        cfg.netIterationLimit = 250;
    case "full"
        cfg.nRepeats = 20;
        cfg.netIterationLimit = 500;
    otherwise
        error('Unknown mode: %s', mode);
end
end

function row = metric_row(repeatNo, seed, responseName, modelName, ...
    selectedKernel, fitSeed, nTrain, nValidation, nTest, y, yhat)
err = yhat - y;
rmse = sqrt(mean(err.^2, 'omitnan'));
mae = mean(abs(err), 'omitnan');
mare = mean(abs(err) ./ max(abs(y), eps), 'omitnan');
ssRes = sum((y-yhat).^2, 'omitnan');
ssTot = sum((y-mean(y, 'omitnan')).^2, 'omitnan');
r2 = 1 - ssRes/max(ssTot, eps);
row = {repeatNo, seed, responseName, char(modelName), char(selectedKernel), ...
    fitSeed, nTrain, nValidation, nTest, rmse, mae, mare, r2};
end

function [scaled, params] = minmax_fit(values)
params.offset = min(values, [], 1);
params.range = max(values, [], 1) - params.offset;
params.range(params.range == 0) = 1;
scaled = (values-params.offset)./params.range;
end

function scaled = minmax_apply(values, params)
scaled = (values-params.offset)./params.range;
end

function values = minmax_reverse(scaled, params)
values = scaled.*params.range + params.offset;
end

function hit = find_file(rootDir, fileName)
projectRoot = fileparts(rootDir);
searchRoots = {rootDir, projectRoot, fullfile(projectRoot, '04_raw_data')};
hits = [];
for iRoot = 1:numel(searchRoots)
    if exist(searchRoots{iRoot}, 'dir')
        hits = [hits; dir(fullfile(searchRoots{iRoot}, '**', fileName))]; %#ok<AGROW>
    end
end
if isempty(hits)
    hit = "";
else
    [~, idx] = min(arrayfun(@(d) strlength(d.folder), hits));
    hit = string(fullfile(hits(idx).folder, hits(idx).name));
end
end
