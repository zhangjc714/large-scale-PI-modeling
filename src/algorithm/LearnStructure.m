function Idx = LearnStructure(Problem,Dataset,params)
% Learn Structural Importance Knowledge

    KeyVarNum = ceil(params.ratio*Problem.D);
    numTree = params.numTree;

    if numel(Dataset.labels) == 0
        % 数据集为空时直接返回随机Idx
        Idx = randperm(Problem.D, KeyVarNum);

    else
        % 训练基于树的分类模型
        ClsModel = fitcensemble(Dataset.EDs, Dataset.labels,'Method','GentleBoost','NumLearningCycles',numTree);
        
        % 获取特征重要性得分
        importance = predictorImportance(ClsModel);
    
        % 获取前x个变量的索引（按重要性降序）
        [~, sortedIdx] = sort(importance, 'descend');
        Idx = sortedIdx(1:KeyVarNum);
    end
end