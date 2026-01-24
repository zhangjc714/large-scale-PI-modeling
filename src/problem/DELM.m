classdef DELM < handle
% Dual Extreme Learning Machine for Interval Prediction
% 
% K-fold cross-valibration training: `obj.train_kfold()`
%     Step 1. 前k-1折训练ELM1点预测模型，在最后1折数据上计算ELM1的点预测输出，并得到残差标签；
%     Step 2. 重复k次，记录所有校准集上的残差输出，构建完整的残差标签，用于ELM2的训练；
%     Step 3. 所有的训练数据输入 + 所有的点预测原始标签：训练ELM1；
%     Step 4. 所有的训练数据输入 + 所有的构建的残差标签：训练ELM2；

    properties
        elm1;   % 第1个ELM：预测均值
        elm2;   % 第2个ELM：预测方差（对数方差）
    end

    methods
        function obj = DELM(w1,b1,w2,b2,lambda)
            obj.elm1 = BasicELM(w1, b1, 'tanh', lambda);
            obj.elm2 = BasicELM(w2, b2, 'tanh', lambda);
        end

        function train_kfold(obj, x, y, k)
            % 训练方法v3：k折交叉训练 ELM1 + ELM2
            if nargin < 4
                k = 5;  % 默认5折
            end
            % 创建k折划分索引
            rng(42);  % 固定随机数种子
            cv = cvpartition(length(y), 'KFold', k);
            e2_all = zeros(size(y));
            % K-Fold交叉验证训练法
            for i = 1:k
                idx_val = test(cv, i);  % 第 i 折作为 ELM2 的训练集（残差估计）
                idx_train = training(cv, i);  % 其他折作为ELM1的训练集（点预测）
                x1 = x(idx_train, :);
                y1 = y(idx_train, :);
                x2 = x(idx_val, :);
                y2 = y(idx_val, :);
                % 训练 ELM1·
                obj.elm1.train(x1, y1);
                % 在第 i 折上计算残差（用于 ELM2 的训练）
                y2_hat = obj.elm1.predict(x2);
                e2 = log((y2 - y2_hat).^2 + eps);  % 残差的对数方差
                % 收集 ELM2 的残差训练数据
                e2_all(idx_val) = e2;
            end
            % 最终训练整合模型：用所有样本重新训练 ELM1 和 ELM2，使它们都基于全数据
            obj.elm1.train(x, y);
            obj.elm2.train(x, e2_all);
        end

        function [yhat_all, e2_all] = get_residual(obj, x, y, k)
            % 获取k折交叉验证构建的残差标签集合（用于展示数据异方差性）
            if nargin < 4
                k = 5;  % 默认5折
            end
            % 创建k折划分索引
            rng(42);  % 固定随机数种子
            cv = cvpartition(length(y), 'KFold', k);
            yhat_all = zeros(size(y));  % 预测值
            e2_all = zeros(size(y));  % 残差标签
            % K-Fold交叉验证训练法
            for i = 1:k
                idx_val = test(cv, i);  % 第 i 折作为 ELM2 的训练集（残差估计）
                idx_train = training(cv, i);  % 其他折作为ELM1的训练集（点预测）
                x1 = x(idx_train, :);
                y1 = y(idx_train, :);
                x2 = x(idx_val, :);
                y2 = y(idx_val, :);
                % 训练 ELM1
                obj.elm1.train(x1, y1);
                % 在第 i 折上计算残差（用于 ELM2 的训练）
                y2_hat = obj.elm1.predict(x2);
                residual = y2 - y2_hat;  % 直接收集残差

                yhat_all(idx_val) = y2_hat;  % 收集预测值
                e2_all(idx_val) = residual;  % 收集 ELM2 的残差训练数据
            end
        end

        function [mu, lb, ub] = predict(obj, x, cl)
            % ===== 第一阶段预测均值 =====
            mu = obj.elm1.predict(x);

            % ===== 第二阶段预测方差 =====
            e = obj.elm2.predict(x);
            std = sqrt(exp(e));                    % 从对数方差恢复标准差

            % ===== 根据置信水平计算区间 =====
            alpha = 1 - cl;
            z = norminv(1 - alpha/2, 0, 1);        % 计算对应的z值

            lb = mu - z .* std ;
            ub = mu + z .* std;
        end
    end

    methods(Static)
        function [PICP, PINAW] = score(y, lb, ub)
            % ===== 区间覆盖率 (PI coverage probability, PICP) =====
            inside = (lb <= y) & (y <= ub);                      % 是否落在区间内
            PICP = mean(inside);                                 % 覆盖率 = 落在区间内的样本比例
            
            % ===== 归一化平均区间宽度 (PI normalized average width, PINAW) =====
            interval_width = ub - lb;                            % 每个样本区间宽度
            PINAW = mean(interval_width) / (max(y) - min(y));    % 平均宽度
        end

        function [RMSE, PICP, PINAW, CWC] = score_all(y, yhat, lb, ub)
            % ===== 均方根误差 (Root Mean Square Error, RMSE) =====
            RMSE = sqrt(mean((y - yhat).^2));

            % ===== 区间覆盖率 (PI coverage probability, PICP) =====
            inside = (lb <= y) & (y <= ub);                      % 是否落在区间内
            PICP = mean(inside);                                 % 覆盖率 = 落在区间内的样本比例
            
            % ===== 归一化平均区间宽度 (PI normalized average width, PINAW) =====
            interval_width = ub - lb;                            % 每个样本区间宽度
            PINAW = mean(interval_width) / (max(y) - min(y));    % 平均宽度

            % ===== 基于覆盖率和宽度的综合性能指标 (Coverage width-based criterion, CWC) =====
            eta = 10;                                            % 宽度与覆盖率的平衡参数
            alpha = 0.05;                                        % 显著性水平，默认 0.05（对应 95% 置信度）
            CWC = (1 - PINAW) .* exp(-eta .* (PICP - (1 - alpha)).^2);
        end
    end
end

