function [OffDec,OffMask] = Operator(Problem,ParentDec,ParentMask,Fitness)
    % The operator of SparseEA

    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------

    %% Parameter setting
    [N,D]       = size(ParentDec);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);
    
    %% 删除错误的检查条件
    % if size(ParentMask, 2) > length(Fitness)
    %     error('Fitness array is too small: %d < %d', length(Fitness), size(ParentMask, 2));
    % end

    %% Crossover for mask
    OffMask = Parent1Mask;
    for i = 1 : N/2
        if rand < 0.5
            index = find(Parent1Mask(i,:) & ~Parent2Mask(i,:));
            if ~isempty(index)
                % 使用父代的适应度进行选择（示例：选择适应度较好的父代）
                if Fitness(i,1) < Fitness(i+N/2,1)
                    selected_index = randi(length(index));
                    OffMask(i,index(selected_index)) = 0;
                end
            end
        else
            index = find(~Parent1Mask(i,:) & Parent2Mask(i,:));
            if ~isempty(index)
                if Fitness(i,1) >= Fitness(i+N/2,1)
                    selected_index = randi(length(index));
                    OffMask(i,index(selected_index)) = Parent2Mask(i,index(selected_index));
                end
            end
        end
    end
    
    %% Mutation for mask
    for i = 1 : N/2
        if rand < 0.5
            index = find(OffMask(i,:));
            if ~isempty(index)
                % 随机变异
                selected_index = randi(length(index));
                OffMask(i,index(selected_index)) = 0;
            end
        else
            index = find(~OffMask(i,:));
            if ~isempty(index)
                selected_index = randi(length(index));
                OffMask(i,index(selected_index)) = 1;
            end
        end
    end
    
    %% Crossover and mutation for dec
    if any(Problem.encoding~=4)
        OffDec = OperatorGAhalf(Problem,ParentDec);
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(N/2,D);
    end
end

function index = TS(Fitness)
    % Binary tournament selection
    if isempty(Fitness)
        index = [];
    else
        index = TournamentSelection(2, 1, Fitness);
        if index > length(Fitness)
            error('Selected index out of bounds: %d > %d', index, length(Fitness));
        end
    end
end