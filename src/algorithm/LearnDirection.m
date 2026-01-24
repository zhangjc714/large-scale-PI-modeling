function ED = LearnDirection(Problem,Dataset,Idx,params)
% Learn Directional Distribution Knowledge

    N = Problem.N;
    d = size(Idx,2);
    
    D = Dataset.EDs(:,Idx);
 
    if size(D,1) <= N
        % 若D中样本数小于概率成分数，则直接返回全为0的ED向量
        learnedED = zeros(N, d);
    else
        if size(D,1) < d
            % 若D中样本数小于特征数，补充零向量使样本数多于特征数d
            D = [D; zeros(d-size(D,1)+1, d)];
        end

        % 初始均值
        mean_init = mean(Dataset.EDs, 1);
        
        % 初始协方差
        cov_init = cov(Dataset.EDs);
        
        % 检查 cov_init 是否是 d x d 的 double 类型
        if ~(isa(cov_init, 'double') && all(size(cov_init) == [d, d]))
            % 如果不是，替换成 d x d 的单位矩阵
            cov_init = eye(d);
        end
        
        % 协方差正定性处理
        C = cov_init(:,:);
        if any(eig(C) <= 0)
            cov_init(:,:) = C + 1e-3 * eye(size(C));
        end

        % 初始化参数
        InitPara = struct('mu', mean_init, 'Sigma', cov_init, 'ComponentProportion', ones(1,N)/N);
        options = statset('MaxIter', params.MaxIter, 'TolFun', params.TolFun);

        try
            % GMM训练
            GenModel = fitgmdist(D, N, 'Start', InitPara, 'Options', options, 'Regularize', params.reg);
        catch
            % 如果产生“病态协方差问题”，则使用 K-means 进行参数初始化
            [~, mean_init] = kmeans(D, N);  % 使用k-means来初始化均值mu
            % 初始协方差矩阵，使用单位协方差
            cov_init = repmat(eye(size(D, 2)), [1, 1, N]);
            % 初始化参数
            InitPara = struct('mu', mean_init, 'Sigma', cov_init, 'ComponentProportion', ones(1, N)/N);
            % GMM训练
            GenModel = fitgmdist(D, N, 'Start', InitPara, 'Options', options, 'Regularize', params.reg);
        end

        learnedED = zeros(N, d);
        for i = 1:N
            % Sample a directional vector
            learnedED(i, :) = mvnrnd(GenModel.mu(i, :), GenModel.Sigma(:, :, i));
        end
    end

    ED = zeros(N, Problem.D);
    ED(:, Idx) = learnedED;
end

