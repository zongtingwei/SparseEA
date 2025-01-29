function [solution, time, off, ofit, site, paretoAVE, tempVar] = SparseEA(train_F, train_L, maxFES, sizep)
    fprintf('SparseEA\n');                                       
    tic;
    global FES;
    FES = 1;
    dim = size(train_F, 2);
    ofit = zeros(sizep, 2); % Objective function values for the population
    initThres = 1;
    thres = 0.1; % Exponential decay constant
    paretoAVE = zeros(1, 2); % To save final result of the Pareto front

    %% Create Problem object
    Problem = struct(...
        'encoding', 4, ...
        'lower', zeros(1, dim), ...
        'upper', ones(1, dim), ...
        'Evaluation', @(Dec) EvaluatePopulation(Dec, train_F, train_L) ...
    );

    %% Initialization
    [Population, Dec, Mask] = InitializePopulation(dim, sizep, Problem.lower, Problem.upper, Problem.encoding, train_F, train_L);

    %% Evaluate initial population
    for i = 1:sizep
        [ofit(i, 1), ofit(i, 2)] = FSKNNfeixiang(Population(i).decs, train_F, train_L);
        if size(ofit(i, :), 2) ~= 2
            error('FSKNNfeixiang returned incorrect Fitness size: %d != 2', size(ofit(i, :), 2));
        end
        Population(i).objs = ofit(i, :);
    end

    %% Environmental selection
    [Population, FrontNo, CrowdDis] = EnvironmentalSelection(Population, sizep);

    %% Optimization
    while FES <= maxFES
        MatingPool = TournamentSelection(2, sizep, FrontNo, -CrowdDis);
        [OffDec, OffMask] = Operator(Problem, Dec(MatingPool, :), Mask(MatingPool, :), ofit(MatingPool, :));
        Offspring = EvaluateOffspring(OffDec, OffMask, train_F, train_L);

        %% Combine current population and offspring
        CombinedPopulation = [Population; Offspring];

        %% Environmental selection for the next generation
        [Population, FrontNo, CrowdDis] = EnvironmentalSelection(CombinedPopulation, sizep);

        %% Update Dec and Mask
        Dec = [Population.decs];
        Mask = [Population.decs]; % Assuming Mask is the same as Dec for simplicity

        %% Evaluate new population
        for i = 1:size(Population, 1)
            [ofit(i, 1), ofit(i, 2)] = FSKNNfeixiang(Population(i).decs, train_F, train_L);
            if size(ofit(i, :), 2) ~= 2
                error('FSKNNfeixiang returned incorrect Fitness size: %d != 2', size(ofit(i, :), 2));
            end
        end

        %% Update solution and pareto front
        site = find(FrontNo == 1);
        solution = ofit(site, :);
        paretoAVE(1) = mean(solution(:, 1));
        paretoAVE(2) = mean(solution(:, 2));

        FES = FES + 1;
    end

    %% Finalization
    tempVar{1} = ofit; % All objective function values
    tempVar{2} = FrontNo; % All front numbers
    tempVar{3} = CrowdDis; % All crowding distances
    tempVar{4} = []; % Other temporary variables if needed

    time = toc;
    off = Dec; % Ensure off is assigned
end

function [Population, Dec, Mask] = InitializePopulation(dim, sizep, lower, upper, encoding, train_F, train_L)
    T = min(dim, sizep * 3);
    PopDec = zeros(sizep, dim); % 决策变量
    PopObj = zeros(sizep, 2); % 假设有两个目标，初始化为0
    PopCon = zeros(sizep, 1); % 假设没有约束违反，初始化为0
    
    for i = 1:sizep
        k = randperm(T, 1);
        j = randperm(dim, k);
        PopDec(i, j) = 1;
    end
    
    % 创建种群对象
    Population = arrayfun(@(i) struct('decs', PopDec(i, :), 'objs', PopObj(i, :), 'cons', PopCon(i)), 1:sizep);
    Dec = PopDec;
    Mask = PopDec; % 初始掩码与决策变量相同
end

function Offspring = EvaluateOffspring(OffDec, OffMask, train_F, train_L)
    % Evaluate the offspring
    N = size(OffDec, 1);
    % 预分配结构体数组（而非cell数组）
    Offspring = repmat(struct('decs', [], 'objs', [], 'cons', []), N, 1);
    for i = 1:N
        [obj1, obj2] = FSKNNfeixiang(OffDec(i, :) .* OffMask(i, :), train_F, train_L);
        Offspring(i).decs = OffDec(i, :);
        Offspring(i).objs = [obj1, obj2];
        Offspring(i).cons = 0;
    end
end

function Population = EvaluatePopulation(Dec, train_F, train_L)
    % Evaluate the population
    N = size(Dec, 1);
    Population = cell(N, 1);
    for i = 1:N
        [obj1, obj2] = FSKNNfeixiang(Dec(i, :), train_F, train_L);
        Population{i} = struct('decs', Dec(i, :), 'objs', [obj1, obj2], 'cons', 0);
    end
end