function run_inverse_baseline_analysis(mode)
%RUN_INVERSE_BASELINE_ANALYSIS Direct macro-to-meso inverse baseline.
%
% This script trains direct inverse surrogates from four macroscopic responses
% to five mesoscopic parameters. It quantifies why the manuscript uses
% forward GPR plus multi-objective search rather than a direct inverse model.

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
meso = table2array(T(:, 1:5));
macro = table2array(T(:, 6:9));
paramNames = {'pb_emod_gpa','pb_kra','pb_ten_mpa','pb_coh_mpa','pb_fa_deg'};

outDir = fullfile(rootDir, "reproduced", "results_inverse_baseline_" + mode);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

checkpointFile = fullfile(outDir, 'inverse_resume_checkpoint.mat');
runsFile = fullfile(outDir, 'inverse_baseline_runs.csv');
runNN = exist('fitrnet', 'file') == 2;
expectedRows = cfg.nRepeats*numel(paramNames)*(1 + double(runNN));
if exist(runsFile, 'file')
    existingRuns = readtable(runsFile, 'VariableNamingRule', 'preserve');
    if height(existingRuns) == expectedRows ...
            && ismember('fit_seed', existingRuns.Properties.VariableNames)
        if exist(checkpointFile, 'file')
            delete(checkpointFile);
        end
        fprintf('Inverse baseline already complete: %s\n', outDir);
        return;
    end
end

records = cell(0, 9);
completedRepeats = 0;
if exist(checkpointFile, 'file')
    saved = load(checkpointFile, 'state');
    state = saved.state;
    if ~isfield(state, 'schemaVersion') || state.schemaVersion ~= 2 ...
            || ~strcmp(state.mode, char(mode)) || state.nRepeats ~= cfg.nRepeats
        error('Incompatible checkpoint: %s', checkpointFile);
    end
    records = state.records;
    completedRepeats = state.completedRepeats;
    fprintf('Resuming inverse baseline after repeat %d/%d.\n', ...
        completedRepeats, cfg.nRepeats);
end
recNo = size(records, 1);
for r = completedRepeats+1:cfg.nRepeats
    repeatSeed = cfg.seedBase + r;
    rng(repeatSeed, 'twister');
    cv = cvpartition(size(macro, 1), 'HoldOut', cfg.holdout);
    idxTrain = training(cv);
    idxTest = test(cv);

    for j = 1:numel(paramNames)
        gprFitSeed = repeatSeed + 10000 + j;
        rng(gprFitSeed, 'twister');
        [xTrain, xScale] = minmax_fit(macro(idxTrain, :));
        xTest = minmax_apply(macro(idxTest, :), xScale);
        [yTrain, yScale] = minmax_fit(meso(idxTrain, j));
        mdl = fitrgp(xTrain, yTrain, ...
            'BasisFunction', 'constant', ...
            'KernelFunction', 'ardrationalquadratic', ...
            'FitMethod', 'sr', ...
            'PredictMethod', 'fic', ...
            'ActiveSetSize', min(1000, size(xTrain, 1)));
        pred = minmax_reverse(predict(mdl, xTest), yScale);
        recNo = recNo + 1;
        records(recNo, :) = metric_row(r, repeatSeed, gprFitSeed, ...
            "inverse_GPR", paramNames{j}, meso(idxTest, j), pred); %#ok<AGROW>

        if runNN
            nnFitSeed = repeatSeed + 20000 + j;
            rng(nnFitSeed, 'twister');
            net = fitrnet(macro(idxTrain, :), meso(idxTrain, j), ...
                'LayerSizes', cfg.netLayerSizes, ...
                'Standardize', true, ...
                'IterationLimit', cfg.netIterationLimit);
            pred = predict(net, macro(idxTest, :));
            recNo = recNo + 1;
            records(recNo, :) = metric_row(r, repeatSeed, nnFitSeed, ...
                "inverse_NN", paramNames{j}, meso(idxTest, j), pred); %#ok<AGROW>
        end
    end
    completedRepeats = r;
    state.schemaVersion = 2;
    state.mode = char(mode);
    state.nRepeats = cfg.nRepeats;
    state.completedRepeats = completedRepeats;
    state.records = records;
    save_checkpoint(checkpointFile, state);
    write_outputs(outDir, records);
    fprintf('Inverse repeat %d/%d complete.\n', r, cfg.nRepeats);
end

write_outputs(outDir, records);
if exist(checkpointFile, 'file')
    delete(checkpointFile);
end

fprintf('Inverse baseline analysis written to %s\n', outDir);
end

function save_checkpoint(checkpointFile, state)
checkpointFile = char(checkpointFile);
tempFile = [checkpointFile '.tmp.mat'];
save(tempFile, 'state', '-v7');
movefile(tempFile, checkpointFile, 'f');
end

function write_outputs(outDir, records)
Runs = cell2table(records, 'VariableNames', ...
    {'repeat','repeat_seed','fit_seed','model','parameter','rmse','mae','mare','r2'});
writetable(Runs, fullfile(outDir, 'inverse_baseline_runs.csv'));

Summary = groupsummary(Runs, {'model','parameter'}, {'mean','std'}, ...
    {'rmse','mae','mare','r2'});
writetable(Summary, fullfile(outDir, 'inverse_baseline_summary.csv'));
end

function cfg = local_config(mode)
cfg.seedBase = 20260710;
cfg.holdout = 0.20;
cfg.netLayerSizes = 10;
switch mode
    case "smoke"
        cfg.nRepeats = 1;
        cfg.netIterationLimit = 50;
    case "quick"
        cfg.nRepeats = 5;
        cfg.netIterationLimit = 150;
    case "review"
        cfg.nRepeats = 10;
        cfg.netIterationLimit = 250;
    case "full"
        cfg.nRepeats = 30;
        cfg.netIterationLimit = 500;
    otherwise
        error('Unknown mode: %s', mode);
end
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

function row = metric_row(repeatNo, repeatSeed, fitSeed, modelName, parameterName, y, yhat)
err = yhat - y;
rmse = sqrt(mean(err.^2, 'omitnan'));
mae = mean(abs(err), 'omitnan');
mare = mean(abs(err) ./ max(abs(y), eps), 'omitnan');
ssRes = sum((y - yhat).^2, 'omitnan');
ssTot = sum((y - mean(y, 'omitnan')).^2, 'omitnan');
r2 = 1 - ssRes / max(ssTot, eps);
row = {repeatNo, repeatSeed, fitSeed, char(modelName), parameterName, ...
    rmse, mae, mare, r2};
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
