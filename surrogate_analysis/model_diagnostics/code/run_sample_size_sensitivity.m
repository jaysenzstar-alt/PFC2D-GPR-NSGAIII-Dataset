function run_sample_size_sensitivity(mode)
%RUN_SAMPLE_SIZE_SENSITIVITY Recompute model accuracy at several data sizes.
%
% Within each repeat, every training size is evaluated on the same 625-case
% test set, and the training subsets are nested. Checkpoints permit restart.

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
responses = {'peak_stress_mpa', 'peak_strain_permille', 'elastic_modulus_gpa', 'poisson_ratio'};

outDir = fullfile(rootDir, "reproduced", "results_sample_size_" + mode);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

checkpointFile = fullfile(outDir, 'sample_size_resume_checkpoint.mat');
runsFile = fullfile(outDir, 'sample_size_repeated_runs.csv');
expectedRows = numel(cfg.trainSizes)*cfg.nRepeats*numel(responses);
if exist(runsFile, 'file')
    existingRuns = readtable(runsFile, 'VariableNamingRule', 'preserve');
    requiredVariables = {'training_size','fit_seed','n_test'};
    if height(existingRuns) == expectedRows ...
            && all(ismember(requiredVariables, existingRuns.Properties.VariableNames))
        if exist(checkpointFile, 'file')
            delete(checkpointFile);
        end
        fprintf('Sample-size sensitivity already complete: %s\n', outDir);
        return;
    end
end

records = cell(0, 11);
completedJobs = zeros(0, 2);
if exist(checkpointFile, 'file')
    saved = load(checkpointFile, 'state');
    state = saved.state;
    if ~isfield(state, 'schemaVersion') || state.schemaVersion ~= 2 ...
            || ~strcmp(state.mode, char(mode)) || state.nRepeats ~= cfg.nRepeats ...
            || ~isequal(state.trainSizes, cfg.trainSizes)
        error('Incompatible checkpoint: %s', checkpointFile);
    end
    records = state.records;
    completedJobs = state.completedJobs;
    fprintf('Resuming sample-size sensitivity with %d/%d jobs complete.\n', ...
        size(completedJobs, 1), numel(cfg.trainSizes)*cfg.nRepeats);
end
recNo = size(records, 1);
n = size(X, 1);
nTest = round(cfg.holdout*n);
for configuredTrainSize = cfg.trainSizes
    trainSize = min(configuredTrainSize, n-nTest);
    for r = 1:cfg.nRepeats
        if ismember([trainSize r], completedJobs, 'rows')
            continue;
        end
        seed = cfg.seedBase + r;
        rng(seed, 'twister');
        order = randperm(n);
        testIdxLocal = order(1:nTest);
        trainingPool = order(nTest+1:end);
        trainIdxLocal = trainingPool(1:trainSize);

        for j = 1:numel(responses)
            fitSeed = seed + 10*trainSize + j;
            rng(fitSeed, 'twister');
            [xTrain, xScale] = minmax_fit(X(trainIdxLocal, :));
            xTest = minmax_apply(X(testIdxLocal, :), xScale);
            [yTrain, yScale] = minmax_fit(Y(trainIdxLocal, j));
            mdl = fitrgp(xTrain, yTrain, ...
                'BasisFunction', 'constant', ...
                'KernelFunction', 'ardrationalquadratic', ...
                'FitMethod', 'sr', ...
                'PredictMethod', 'fic', ...
                'ActiveSetSize', min(1000, size(xTrain, 1)));
            yPred = minmax_reverse(predict(mdl, xTest), yScale);
            recNo = recNo + 1;
            records(recNo, :) = metric_row(trainSize, r, seed, fitSeed, nTest, ...
                responses{j}, Y(testIdxLocal, j), yPred); %#ok<AGROW>
        end
        completedJobs(end+1, :) = [trainSize r]; %#ok<AGROW>
        state.schemaVersion = 2;
        state.mode = char(mode);
        state.nRepeats = cfg.nRepeats;
        state.trainSizes = cfg.trainSizes;
        state.completedJobs = completedJobs;
        state.records = records;
        save_checkpoint(checkpointFile, state);
        write_outputs(outDir, records);
        fprintf('Training size %d, repeat %d/%d complete (%d/%d jobs).\n', ...
            trainSize, r, cfg.nRepeats, size(completedJobs, 1), ...
            numel(cfg.trainSizes)*cfg.nRepeats);
    end
end

write_outputs(outDir, records);
if exist(checkpointFile, 'file')
    delete(checkpointFile);
end

fprintf('Sample-size sensitivity written to %s\n', outDir);
end

function save_checkpoint(checkpointFile, state)
checkpointFile = char(checkpointFile);
tempFile = [checkpointFile '.tmp.mat'];
save(tempFile, 'state', '-v7');
movefile(tempFile, checkpointFile, 'f');
end

function write_outputs(outDir, records)
Runs = cell2table(records, 'VariableNames', ...
    {'training_size','repeat','seed','fit_seed','n_train','n_test','response', ...
    'rmse','mae','mare','r2'});
writetable(Runs, fullfile(outDir, 'sample_size_repeated_runs.csv'));

Summary = groupsummary(Runs, {'training_size','n_test','response'}, {'mean','std'}, ...
    {'rmse','mae','mare','r2'});
writetable(Summary, fullfile(outDir, 'sample_size_summary.csv'));
end

function cfg = local_config(mode)
cfg.seedBase = 20260710;
cfg.holdout = 0.20;
switch mode
    case "smoke"
        cfg.trainSizes = [160 400];
        cfg.nRepeats = 1;
    case "quick"
        cfg.trainSizes = [400 800 1200 2000 2500];
        cfg.nRepeats = 3;
    case "review"
        cfg.trainSizes = [400 800 1200 1600 2000 2500];
        cfg.nRepeats = 5;
    case "full"
        cfg.trainSizes = [200 400 600 800 1200 1600 2000 2500];
        cfg.nRepeats = 20;
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

function row = metric_row(trainSize, repeatNo, seed, fitSeed, nTest, responseName, y, yhat)
err = yhat - y;
rmse = sqrt(mean(err.^2, 'omitnan'));
mae = mean(abs(err), 'omitnan');
mare = mean(abs(err) ./ max(abs(y), eps), 'omitnan');
ssRes = sum((y - yhat).^2, 'omitnan');
ssTot = sum((y - mean(y, 'omitnan')).^2, 'omitnan');
r2 = 1 - ssRes / max(ssTot, eps);
row = {trainSize, repeatNo, seed, fitSeed, trainSize, nTest, responseName, ...
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
