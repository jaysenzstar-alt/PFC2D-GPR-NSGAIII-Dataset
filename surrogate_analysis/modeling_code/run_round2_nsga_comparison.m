function run_round2_nsga_comparison(mode, modelFile, outputDir)
%RUN_ROUND2_NSGA_COMPARISON Compare NSGA-II and NSGA-III using four GPR objectives.
%
% run_round2_nsga_comparison('review') uses the archived production models.
% Pass a model file explicitly to use newly fitted models:
%   run_round2_nsga_comparison('review', modelFile, outputDir)
%
% Modes: smoke (one target/run); quick (three runs); review (reported 60 x 30
% search budget, ten runs); full (exploratory 150 x 80 budget, ten runs).
% Outputs include per-run fronts, run metrics, summaries and a run manifest.
% The output directory must be absent or empty.
% Requires MATLAB Statistics and Machine Learning Toolbox.

if nargin < 1 || isempty(mode)
    mode = 'quick';
end
mode = lower(string(mode));

clc;

cfg = default_config(mode);
if nargin >= 2 && ~isempty(modelFile)
    cfg.modelFile = char(modelFile);
end
if nargin >= 3 && ~isempty(outputDir)
    cfg.resultsDir = char(outputDir);
end
ensure_empty_output_dir(cfg.resultsDir);

models = load_surrogate_models(cfg.modelFile);
targets = readtable(cfg.targetsCsv);

allRuns = {};
runCount = 0;

fprintf('Running comparison for %d target cases...\n', height(targets));

for iTarget = 1:min(height(targets), cfg.sampleLimit)
    target = [
        targets.target_peak_stress_mpa(iTarget), ...
        targets.target_peak_strain_permille(iTarget), ...
        targets.target_elastic_modulus_gpa(iTarget), ...
        targets.target_poisson_ratio(iTarget)
    ];
    sampleId = targets.sample(iTarget);

    fprintf('\nTarget sample %d: [%g %g %g %g]\n', sampleId, target);

    for iAlg = 1:numel(cfg.algorithms)
        algorithm = cfg.algorithms{iAlg};

        for iRun = 1:cfg.nRuns
            seed = cfg.seedBase + 10000*iTarget + 100*iAlg + iRun;
            rng(seed, 'twister');
            tStart = tic;

            objective = @(x) objective_relative_error(x, target, models);
            switch upper(algorithm)
                case 'NSGAII'
                    pop = run_nsga2(objective, cfg);
                case 'NSGAIII'
                    pop = run_nsga3(objective, cfg);
                otherwise
                    error('Unknown algorithm: %s', algorithm);
            end

            elapsed = toc(tStart);
            front = first_front(pop);
            [bestMean, bestIdx] = min(mean(cat(1, front.Cost), 2));
            best = front(bestIdx);

            runCount = runCount + 1;
            rec.sample = sampleId;
            rec.target = target;
            rec.algorithm = algorithm;
            rec.run = iRun;
            rec.seed = seed;
            rec.elapsed_sec = elapsed;
            rec.front = front;
            rec.best_mean_error = bestMean;
            rec.best_position = best.Position;
            rec.best_cost = best.Cost;
            rec.igd = NaN;
            rec.hypervolume = NaN;
            allRuns{runCount, 1} = rec; %#ok<AGROW>

            paretoPath = fullfile(cfg.resultsDir, sprintf('pareto_sample%02d_%s_run%02d.csv', sampleId, algorithm, iRun));
            write_pareto_csv(paretoPath, sampleId, algorithm, iRun, target, front);

            fprintf('  %-7s run %02d | F1=%3d | best mean error=%.6f | %.2fs\n', ...
                algorithm, iRun, numel(front), bestMean, elapsed);
        end
    end
end

allRuns = compute_quality_metrics(allRuns, cfg);
write_run_table(allRuns, cfg);
write_summary_table(allRuns, cfg);
write_configuration_manifest(cfg, height(targets));

fprintf('\nDone. Results written to: %s\n', cfg.resultsDir);

end


function cfg = default_config(mode)
codeDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(codeDir);
inputDir = fullfile(rootDir, 'input_data');
modelDir = fullfile(rootDir, 'trained_models');
resultsDir = fullfile(rootDir, 'modeling_results');

cfg.modelFile = fullfile(modelDir, ...
    'round2_production_models.mat');
cfg.targetsCsv = fullfile(inputDir, 'targets_12.csv');
cfg.resultsDir = char(fullfile(resultsDir, 'reproduced', "optimizer_" + mode));

cfg.algorithms = {'NSGAII', 'NSGAIII'};
cfg.seedBase = 20260715;
cfg.sampleLimit = inf;

cfg.nVar = 5;
cfg.varMin = [15, 1, 15, 15, 15];
cfg.varMax = [75, 3, 95, 35, 75];

switch mode
    case "smoke"
        cfg.nRuns = 1;
        cfg.nPop = 20;
        cfg.maxIt = 5;
        cfg.hypervolumeSamples = 5000;
        cfg.sampleLimit = 1;
    case "quick"
        cfg.nRuns = 3;
        cfg.nPop = 60;
        cfg.maxIt = 30;
        cfg.hypervolumeSamples = 50000;
    case "review"
        % Reported controlled benchmark: 60 individuals, 30 generations.
        cfg.nRuns = 10;
        cfg.nPop = 60;
        cfg.maxIt = 30;
        cfg.hypervolumeSamples = 100000;
    case "full"
        % Optional larger search; not the reported controlled benchmark.
        cfg.nRuns = 10;
        cfg.nPop = 150;
        cfg.maxIt = 80;
        cfg.hypervolumeSamples = 200000;
    otherwise
        error('Unknown mode: %s. Use smoke, quick, review, or full.', mode);
end

cfg.pCrossover = 0.5;
cfg.pMutation = 0.5;
cfg.mu = 0.02; % Coordinate-count factor: ceil(0.02*5) = 1, not a per-variable probability.
cfg.sigma = 0.1*(cfg.varMax-cfg.varMin);
cfg.nDivision = 10;

% Relative-error objectives are normally much smaller than 1.0. The reference
% point is kept at 1.0 for all objectives and dominated points are clipped for
% Monte-Carlo HV estimation.
cfg.hypervolumeRef = [1, 1, 1, 1];
end


function models = load_surrogate_models(modelFile)
if ~isfile(modelFile)
    error(['Missing production model bundle: %s. Supply a valid model file ' ...
        'as the second function argument.'], modelFile);
end
raw = load(modelFile, 'modelBundle');
if ~isfield(raw, 'modelBundle')
    error('Expected variable modelBundle in %s.', modelFile);
end
models = raw.modelBundle;
if ~isfield(models, 'schemaVersion') || models.schemaVersion ~= 1
    error('Unsupported model-bundle schema in %s.', modelFile);
end
required = {'models','xScale','yScales','varMin','varMax','responseNames'};
for i = 1:numel(required)
    if ~isfield(models, required{i})
        error('Model bundle is missing field %s.', required{i});
    end
end
if numel(models.models) ~= 4 || numel(models.yScales) ~= 4
    error('Expected four production GPR models and four response scalers.');
end
end


function cost = objective_relative_error(position, target, models)
prediction = surrogate_predict(position, models);
den = max(abs(target), eps);
cost = abs(prediction - target)./den;
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


function pop = run_nsga2(objective, cfg)
pop = initialize_population(objective, cfg);
pop = assign_rank_and_crowding(pop);

nCrossover = 2*round(cfg.pCrossover*cfg.nPop/2);
nMutation = round(cfg.pMutation*cfg.nPop);

for it = 1:cfg.maxIt %#ok<NASGU>
    offspring = empty_individual(nCrossover+nMutation);
    idx = 0;

    for k = 1:2:nCrossover
        p1 = tournament_select_nsga2(pop);
        p2 = tournament_select_nsga2(pop);
        [x1, x2] = crossover_arithmetic(p1.Position, p2.Position);
        idx = idx + 1;
        offspring(idx) = make_individual(bound_position(x1, cfg), objective);
        idx = idx + 1;
        offspring(idx) = make_individual(bound_position(x2, cfg), objective);
    end

    for k = 1:nMutation
        p = tournament_select_nsga2(pop);
        x = mutate_gaussian(p.Position, cfg);
        idx = idx + 1;
        offspring(idx) = make_individual(bound_position(x, cfg), objective);
    end

    combined = [pop; offspring(:)];
    combined = assign_rank_and_crowding(combined);
    pop = environmental_select_nsga2(combined, cfg.nPop);
end
end


function pop = run_nsga3(objective, cfg)
pop = initialize_population(objective, cfg);
refPoints = generate_reference_points(4, cfg.nDivision);
pop = assign_rank(pop);

nCrossover = 2*round(cfg.pCrossover*cfg.nPop/2);
nMutation = round(cfg.pMutation*cfg.nPop);

for it = 1:cfg.maxIt %#ok<NASGU>
    offspring = empty_individual(nCrossover+nMutation);
    idx = 0;

    for k = 1:2:nCrossover
        p1 = tournament_select_by_rank(pop);
        p2 = tournament_select_by_rank(pop);
        [x1, x2] = crossover_arithmetic(p1.Position, p2.Position);
        idx = idx + 1;
        offspring(idx) = make_individual(bound_position(x1, cfg), objective);
        idx = idx + 1;
        offspring(idx) = make_individual(bound_position(x2, cfg), objective);
    end

    for k = 1:nMutation
        p = tournament_select_by_rank(pop);
        x = mutate_gaussian(p.Position, cfg);
        idx = idx + 1;
        offspring(idx) = make_individual(bound_position(x, cfg), objective);
    end

    combined = [pop; offspring(:)];
    pop = environmental_select_nsga3(combined, cfg.nPop, refPoints);
end
end


function pop = initialize_population(objective, cfg)
pop = empty_individual(cfg.nPop);
for i = 1:cfg.nPop
    x = cfg.varMin + rand(1, cfg.nVar).*(cfg.varMax-cfg.varMin);
    pop(i) = make_individual(x, objective);
end
end


function pop = empty_individual(n)
template.Position = [];
template.Cost = [];
template.Rank = [];
template.CrowdingDistance = [];
template.DominationSet = [];
template.DominatedCount = [];
pop = repmat(template, n, 1);
end


function ind = make_individual(position, objective)
ind.Position = position;
ind.Cost = objective(position);
ind.Rank = [];
ind.CrowdingDistance = [];
ind.DominationSet = [];
ind.DominatedCount = [];
end


function x = bound_position(x, cfg)
x = max(x, cfg.varMin);
x = min(x, cfg.varMax);
end


function [y1, y2] = crossover_arithmetic(x1, x2)
alpha = rand(size(x1));
y1 = alpha.*x1 + (1-alpha).*x2;
y2 = alpha.*x2 + (1-alpha).*x1;
end


function y = mutate_gaussian(x, cfg)
y = x;
nVar = numel(x);
nMu = max(1, ceil(cfg.mu*nVar));
j = randperm(nVar, nMu);
y(j) = x(j) + cfg.sigma(j).*randn(size(j));
end


function selected = tournament_select_nsga2(pop)
i = randi(numel(pop));
j = randi(numel(pop));
a = pop(i);
b = pop(j);
if a.Rank < b.Rank
    selected = a;
elseif b.Rank < a.Rank
    selected = b;
elseif a.CrowdingDistance > b.CrowdingDistance
    selected = a;
else
    selected = b;
end
end


function selected = tournament_select_by_rank(pop)
i = randi(numel(pop));
j = randi(numel(pop));
a = pop(i);
b = pop(j);
if a.Rank < b.Rank
    selected = a;
elseif b.Rank < a.Rank
    selected = b;
else
    if rand < 0.5
        selected = a;
    else
        selected = b;
    end
end
end


function pop = assign_rank_and_crowding(pop)
[pop, fronts] = assign_rank(pop);
for iFront = 1:numel(fronts)
    pop = assign_crowding_distance(pop, fronts{iFront});
end
end


function [pop, fronts] = assign_rank(pop)
nPop = numel(pop);
fronts = {};
fronts{1} = [];

for i = 1:nPop
    pop(i).DominationSet = [];
    pop(i).DominatedCount = 0;
end

for i = 1:nPop
    for j = i+1:nPop
        if dominates(pop(i).Cost, pop(j).Cost)
            pop(i).DominationSet = [pop(i).DominationSet, j]; %#ok<AGROW>
            pop(j).DominatedCount = pop(j).DominatedCount + 1;
        elseif dominates(pop(j).Cost, pop(i).Cost)
            pop(j).DominationSet = [pop(j).DominationSet, i]; %#ok<AGROW>
            pop(i).DominatedCount = pop(i).DominatedCount + 1;
        end
    end
    if pop(i).DominatedCount == 0
        pop(i).Rank = 1;
        fronts{1} = [fronts{1}, i]; %#ok<AGROW>
    end
end

k = 1;
while true
    nextFront = [];
    for i = fronts{k}
        for j = pop(i).DominationSet
            pop(j).DominatedCount = pop(j).DominatedCount - 1;
            if pop(j).DominatedCount == 0
                pop(j).Rank = k + 1;
                nextFront = [nextFront, j]; %#ok<AGROW>
            end
        end
    end
    if isempty(nextFront)
        break;
    end
    fronts{k+1} = nextFront; %#ok<AGROW>
    k = k + 1;
end
end


function flag = dominates(a, b)
flag = all(a <= b) && any(a < b);
end


function pop = assign_crowding_distance(pop, frontIdx)
if isempty(frontIdx)
    return;
end
nFront = numel(frontIdx);
costs = cat(1, pop(frontIdx).Cost);
nObj = size(costs, 2);
distance = zeros(nFront, 1);

for j = 1:nObj
    [sortedCost, order] = sort(costs(:, j));
    distance(order(1)) = inf;
    distance(order(end)) = inf;
    denom = sortedCost(end) - sortedCost(1);
    if denom <= eps
        continue;
    end
    for k = 2:nFront-1
        distance(order(k)) = distance(order(k)) + (sortedCost(k+1)-sortedCost(k-1))/denom;
    end
end

for k = 1:nFront
    pop(frontIdx(k)).CrowdingDistance = distance(k);
end
end


function selected = environmental_select_nsga2(pop, nPop)
[~, order] = sortrows([[pop.Rank]', -[pop.CrowdingDistance]']);
pop = pop(order);
selected = pop(1:nPop);
end


function selected = environmental_select_nsga3(pop, nPop, refPoints)
[pop, fronts] = assign_rank(pop);
selectedIdx = [];
lastFront = [];

for iFront = 1:numel(fronts)
    front = fronts{iFront};
    if numel(selectedIdx) + numel(front) <= nPop
        selectedIdx = [selectedIdx, front]; %#ok<AGROW>
    else
        lastFront = front;
        break;
    end
end

if numel(selectedIdx) < nPop
    need = nPop - numel(selectedIdx);
    chosenLast = select_by_reference_points(pop, selectedIdx, lastFront, refPoints, need);
    selectedIdx = [selectedIdx, chosenLast]; %#ok<AGROW>
end

selected = pop(selectedIdx);
end


function chosen = select_by_reference_points(pop, selectedIdx, candidateIdx, refPoints, need)
allIdx = [selectedIdx, candidateIdx];
costs = cat(1, pop(allIdx).Cost);
zmin = min(costs, [], 1);
zmax = max(costs, [], 1);
normCosts = (costs - zmin)./max(zmax-zmin, eps);

assoc = zeros(numel(allIdx), 1);
dist = zeros(numel(allIdx), 1);
for i = 1:numel(allIdx)
    [assoc(i), dist(i)] = nearest_reference(normCosts(i, :), refPoints);
end

nicheCount = zeros(size(refPoints, 1), 1);
for i = 1:numel(selectedIdx)
    nicheCount(assoc(i)) = nicheCount(assoc(i)) + 1;
end

candidateLocalStart = numel(selectedIdx) + 1;
candidateLocal = candidateLocalStart:numel(allIdx);
available = true(numel(candidateIdx), 1);
chosen = [];

while numel(chosen) < need && any(available)
    availableLocal = candidateLocal(available);
    refs = unique(assoc(availableLocal));
    counts = nicheCount(refs);
    minCount = min(counts);
    minRefs = refs(counts == minCount);
    ref = minRefs(randi(numel(minRefs)));

    localForRef = availableLocal(assoc(availableLocal) == ref);
    if nicheCount(ref) == 0
        [~, bestLocal] = min(dist(localForRef));
        chosenLocal = localForRef(bestLocal);
    else
        chosenLocal = localForRef(randi(numel(localForRef)));
    end

    globalIdx = allIdx(chosenLocal);
    chosen = [chosen, globalIdx]; %#ok<AGROW>
    nicheCount(ref) = nicheCount(ref) + 1;
    available(chosenLocal - candidateLocalStart + 1) = false;
end
end


function [idx, d] = nearest_reference(point, refPoints)
normRef = sqrt(sum(refPoints.^2, 2));
normRef(normRef == 0) = eps;
projection = (refPoints*point(:))./(normRef.^2);
closest = projection.*refPoints;
distances = sqrt(sum((closest - point).^2, 2));
[d, idx] = min(distances);
end


function refPoints = generate_reference_points(nObj, nDiv)
refPoints = [];
current = zeros(1, nObj);
refPoints = recursive_ref_points(refPoints, current, nObj, nDiv, nDiv, 1);
end


function refPoints = recursive_ref_points(refPoints, current, nObj, left, total, idx)
if idx == nObj
    current(idx) = left/total;
    refPoints = [refPoints; current]; %#ok<AGROW>
else
    for i = 0:left
        current(idx) = i/total;
        refPoints = recursive_ref_points(refPoints, current, nObj, left-i, total, idx+1);
    end
end
end


function front = first_front(pop)
[pop, fronts] = assign_rank(pop);
front = pop(fronts{1});
end


function allRuns = compute_quality_metrics(allRuns, cfg)
samples = unique(cellfun(@(r) r.sample, allRuns));
for iSample = 1:numel(samples)
    sample = samples(iSample);
    idx = find(cellfun(@(r) r.sample == sample, allRuns));
    unionCosts = [];
    for k = idx(:)'
        unionCosts = [unionCosts; cat(1, allRuns{k}.front.Cost)]; %#ok<AGROW>
    end
    referencePF = nondominated_rows(unionCosts);

    for k = idx(:)'
        costs = cat(1, allRuns{k}.front.Cost);
        allRuns{k}.igd = igd_metric(costs, referencePF);
        allRuns{k}.hypervolume = hypervolume_mc(costs, cfg.hypervolumeRef, cfg.hypervolumeSamples, allRuns{k}.seed);
    end
end
end


function pf = nondominated_rows(costs)
n = size(costs, 1);
keep = true(n, 1);
for i = 1:n
    if ~keep(i)
        continue;
    end
    for j = 1:n
        if i ~= j && dominates(costs(j, :), costs(i, :))
            keep(i) = false;
            break;
        end
    end
end
pf = costs(keep, :);
end


function score = igd_metric(popObj, referencePF)
distances = zeros(size(referencePF, 1), 1);
for i = 1:size(referencePF, 1)
    d = sqrt(sum((popObj - referencePF(i, :)).^2, 2));
    distances(i) = min(d);
end
score = mean(distances);
end


function hv = hypervolume_mc(popObj, refPoint, nSamples, seed)
% Monte-Carlo hypervolume estimate for minimization. This is robust for four
% objectives and avoids toolbox-specific exact-HV implementations.
rng(seed + 777, 'twister');
popObj = min(max(popObj, 0), refPoint);
if isempty(popObj)
    hv = 0;
    return;
end
samples = rand(nSamples, numel(refPoint)).*refPoint;
dominated = false(nSamples, 1);
for i = 1:size(popObj, 1)
    dominated = dominated | all(samples >= popObj(i, :), 2);
end
hv = prod(refPoint) * mean(dominated);
end


function write_pareto_csv(path, sample, algorithm, runId, target, front)
n = numel(front);
sampleCol = repmat(sample, n, 1);
algorithmCol = repmat({algorithm}, n, 1);
runCol = repmat(runId, n, 1);
targetMat = repmat(target, n, 1);
pos = cat(1, front.Position);
cost = cat(1, front.Cost);

T = table(sampleCol, algorithmCol, runCol, ...
    targetMat(:, 1), targetMat(:, 2), targetMat(:, 3), targetMat(:, 4), ...
    pos(:, 1), pos(:, 2), pos(:, 3), pos(:, 4), pos(:, 5), ...
    cost(:, 1), cost(:, 2), cost(:, 3), cost(:, 4), mean(cost, 2), ...
    'VariableNames', {'sample','algorithm','run', ...
    'target_peak_stress_mpa','target_peak_strain_permille','target_elastic_modulus_gpa','target_poisson_ratio', ...
    'pb_emod_gpa','pb_kra','pb_ten_mpa','pb_coh_mpa','pb_fa_deg', ...
    'err_peak_stress','err_peak_strain','err_elastic_modulus','err_poisson_ratio','mean_relative_error'});
writetable(T, path);
end


function write_run_table(allRuns, cfg)
n = numel(allRuns);
sample = zeros(n, 1);
algorithm = cell(n, 1);
runId = zeros(n, 1);
seed = zeros(n, 1);
elapsed = zeros(n, 1);
nFront = zeros(n, 1);
bestMean = zeros(n, 1);
bestPos = zeros(n, 5);
bestCost = zeros(n, 4);
igd = zeros(n, 1);
hv = zeros(n, 1);

for i = 1:n
    r = allRuns{i};
    sample(i) = r.sample;
    algorithm{i} = r.algorithm;
    runId(i) = r.run;
    seed(i) = r.seed;
    elapsed(i) = r.elapsed_sec;
    nFront(i) = numel(r.front);
    bestMean(i) = r.best_mean_error;
    bestPos(i, :) = r.best_position;
    bestCost(i, :) = r.best_cost;
    igd(i) = r.igd;
    hv(i) = r.hypervolume;
end

T = table(sample, algorithm, runId, seed, elapsed, nFront, bestMean, ...
    bestPos(:,1), bestPos(:,2), bestPos(:,3), bestPos(:,4), bestPos(:,5), ...
    bestCost(:,1), bestCost(:,2), bestCost(:,3), bestCost(:,4), igd, hv, ...
    'VariableNames', {'sample','algorithm','run','seed','elapsed_sec','front_size','best_mean_relative_error', ...
    'best_pb_emod_gpa','best_pb_kra','best_pb_ten_mpa','best_pb_coh_mpa','best_pb_fa_deg', ...
    'best_err_peak_stress','best_err_peak_strain','best_err_elastic_modulus','best_err_poisson_ratio','igd','hypervolume'});

writetable(T, fullfile(cfg.resultsDir, 'nsga_comparison_runs.csv'));
end


function write_summary_table(allRuns, cfg)
runFile = fullfile(cfg.resultsDir, 'nsga_comparison_runs.csv');
T = readtable(runFile);
groups = unique(T(:, {'sample', 'algorithm'}), 'rows');

rows = {};
for i = 1:height(groups)
    mask = T.sample == groups.sample(i) & strcmp(T.algorithm, groups.algorithm{i});
    S = T(mask, :);
    row.sample = groups.sample(i);
    row.algorithm = groups.algorithm{i};
    row.n_runs = height(S);
    row.mean_best_error = mean(S.best_mean_relative_error);
    row.std_best_error = std(S.best_mean_relative_error);
    row.mean_igd = mean(S.igd);
    row.std_igd = std(S.igd);
    row.mean_hypervolume = mean(S.hypervolume);
    row.std_hypervolume = std(S.hypervolume);
    row.mean_elapsed_sec = mean(S.elapsed_sec);
    rows{end+1, 1} = row; %#ok<AGROW>
end

sample = cellfun(@(r) r.sample, rows);
algorithm = cellfun(@(r) r.algorithm, rows, 'UniformOutput', false);
n_runs = cellfun(@(r) r.n_runs, rows);
mean_best_error = cellfun(@(r) r.mean_best_error, rows);
std_best_error = cellfun(@(r) r.std_best_error, rows);
mean_igd = cellfun(@(r) r.mean_igd, rows);
std_igd = cellfun(@(r) r.std_igd, rows);
mean_hypervolume = cellfun(@(r) r.mean_hypervolume, rows);
std_hypervolume = cellfun(@(r) r.std_hypervolume, rows);
mean_elapsed_sec = cellfun(@(r) r.mean_elapsed_sec, rows);

Summary = table(sample(:), algorithm(:), n_runs(:), mean_best_error(:), std_best_error(:), ...
    mean_igd(:), std_igd(:), mean_hypervolume(:), std_hypervolume(:), mean_elapsed_sec(:), ...
    'VariableNames', {'sample','algorithm','n_runs','mean_best_error','std_best_error', ...
    'mean_igd','std_igd','mean_hypervolume','std_hypervolume','mean_elapsed_sec'});

writetable(Summary, fullfile(cfg.resultsDir, 'nsga_comparison_summary.csv'));
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


function write_configuration_manifest(cfg, targetCount)
path = fullfile(cfg.resultsDir, 'RUN_MANIFEST.txt');
fid = fopen(path, 'wt');
if fid < 0
    error('Could not create manifest: %s', path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'script=%s\n', mfilename);
fprintf(fid, 'matlab_version=%s\n', version);
fprintf(fid, 'model_file=%s\n', cfg.modelFile);
fprintf(fid, 'targets_csv=%s\n', cfg.targetsCsv);
fprintf(fid, 'target_count=%d\n', min(targetCount, cfg.sampleLimit));
fprintf(fid, 'algorithms=%s\n', strjoin(cfg.algorithms, ','));
fprintf(fid, 'n_runs=%d\n', cfg.nRuns);
fprintf(fid, 'n_population=%d\n', cfg.nPop);
fprintf(fid, 'max_iterations=%d\n', cfg.maxIt);
fprintf(fid, 'seed_base=%d\n', cfg.seedBase);
fprintf(fid, 'mutation_coordinate_count=%d\n', max(1, ceil(cfg.mu*cfg.nVar)));
fprintf(fid, 'mutation_step_fraction=0.1\n');
fprintf(fid, 'bounds_min=%s\n', mat2str(cfg.varMin));
fprintf(fid, 'bounds_max=%s\n', mat2str(cfg.varMax));
fprintf(fid, 'hypervolume_reference=%s\n', mat2str(cfg.hypervolumeRef));
fprintf(fid, 'hypervolume_samples=%d\n', cfg.hypervolumeSamples);
fprintf(fid, 'igd_reference=pooled nondominated union of both algorithms and all runs for each target\n');
fprintf(fid, 'boundary_policy=repair every offspring to the reviewed training-domain bounds\n');
end
