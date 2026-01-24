classdef IPMOP < PROBLEM
% Interval Prediction Multiobjective Optimization Problem (IPMOP) 
% 大规模多目标预测区间建模优化问题
% 
% 编码方案-分为四个部分：
%     + ELM1的权重 (实数，1个矩阵展开)
%     + ELM1的偏置 (实数，1行)
%     + ELM2的权重 (实数，1个矩阵展开)
%     + ELM2的偏置 (实数，1行)
% 
% 解的表现型pheno（n为输入特征数，h1和h2分别为两个ELM的隐层节点数）：
%     + w1: [n × h1] 矩阵
%     + b1: [1 × h1] 行向量
%     + w2: [n × h2] 矩阵
%     + b2: [1 × h2] 行向量
% 
% 解的基因型geno：
%     [(n+1) × (h1+h2)] 行向量

    properties(SetAccess = private)
        n;         % 输入特征数
        h1;        % ELM1隐层节点数
        h2;        % ELM2隐层节点数
        cl;        % 置信水平
        lambda;    % ELM正则化参数
        data;      % 数据集
    end

    methods
        %% Default settings of the problem
        function Setting(obj)
            % 加载超参数
            params = SetParams();
            obj.h1 = params.h1;  % ELM1隐层节点数
            obj.h2 = params.h2;  % ELM2隐层节点数
            obj.cl = params.cl;  % 置信水平
            obj.lambda = params.lambda;  % ELM正则化参数
            % 加载训练和验证数据
            obj.data = LoadData("train_val_test");  % kfold训练
            % 输入特征数
            obj.n = size(obj.data.train_x, 2);
            % 决策变量上下界
            [obj.lower,obj.upper] = IPMOP.SetLowerUpperBounds(obj.n,obj.h1,obj.h2);
            % 编码方式
            obj.encoding = [ones(1, (obj.n+1)*(obj.h1+obj.h2))];  
            % 目标维度、变量维度
            obj.M = 2;
            obj.D = size(obj.encoding,2);
        end

        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            PopObj = zeros(size(PopDec,1), 2);
            for i = 1 : size(PopDec,1)
                % 获取表现型
                pheno = IPMOP.Decode(PopDec(i,:), obj.n, obj.h1, obj.h2);
                % 构建DELM区间预测模型
                PImodel = DELM(pheno.w1, pheno.b1, pheno.w2, pheno.b2, obj.lambda);
                % 使用训练集训练模型
                PImodel.train_kfold(obj.data.train_x, obj.data.train_y);  % kfold训练
                % 获取验证集上的预测区间
                [~, lb, ub] = PImodel.predict(obj.data.val_x, obj.cl);
                % 计算区间覆盖率和区间宽度
                [PICP, PINAW] = PImodel.score(obj.data.val_y, lb, ub);
                % 目标函数赋值
                PopObj(i,:) = [-PICP, PINAW];  % 最大化覆盖率 & 最小化区间宽度
            end
        end
        function R = GetOptimum(obj,N)
            % 计算HV指标，需要设置nadir point
            % f1：覆盖率最差为0%；
            % f2：区间宽度最差是1.0，因为硅含量标签是归一化到0~1区间的，区间宽度最宽为1（硅含量值的变化范围）；
            R = [0, 1];
        end
    end

    methods(Static)
        function [lb,ub] = SetLowerUpperBounds(n,h1,h2)
        % 输入：
        %     n:  输入特征数
        %     h1: ELM1的隐层节点数
        %     h2: ELM2的隐层节点数
        % 输出：
        %     lb: 决策变量下界
        %     ub: 决策变量上界
            % 下界定义
            lb_pheno = {};
            lb_pheno.w1 = repmat(-sqrt(6/(n+h1)), n, h1);
            lb_pheno.b1 = repmat(-0.1, 1, h1);
            lb_pheno.w2 = repmat(-sqrt(6/(n+h2)), n, h2);
            lb_pheno.b2 = repmat(-0.1, 1, h2);
            % 上界定义
            ub_pheno = {};
            ub_pheno.w1 = repmat(sqrt(6/(n+h1)), n, h1);
            ub_pheno.b1 = repmat(0.1, 1, h1);
            ub_pheno.w2 = repmat(sqrt(6/(n+h2)), n, h2);
            ub_pheno.b2 = repmat(0.1, 1, h2);
            % 解码得到上下界一维向量
            lb = IPMOP.Encode(lb_pheno);
            ub = IPMOP.Encode(ub_pheno);
        end

        function geno = Encode(pheno)
        % 解的编码（表现型 -> 基因型）
        % 输入：pheno：包含4个部分的结构体（ELM1权重、ELM1偏置、ELM2权重、ELM2偏置）
        % 输出：geno：决策变量向量（一维行向量）
            geno = [pheno.w1(:)', pheno.b1, pheno.w2(:)', pheno.b2];
        end

        function pheno = Decode(geno,n,h1,h2)
        % 解的解码（基因型 -> 表现型）
        % 输入：geno：决策变量向量（一维行向量）
        %       n:  输入特征数
        %       h1: ELM1的隐层节点数
        %       h2: ELM2的隐层节点数
        % 输出：pheno：包含4个部分的结构体（ELM1权重、ELM1偏置、ELM2权重、ELM2偏置）
            % 分割点
            seg1 = n * h1;
            seg2 = seg1 + h1;
            seg3 = seg2 + n * h2;
            % 表现型
            pheno.w1 = reshape(geno(1:seg1),[n,h1]);
            pheno.b1 = geno(seg1+1:seg2);
            pheno.w2 = reshape(geno(seg2+1:seg3),[n,h2]);
            pheno.b2 = geno(seg3+1:end);
        end
    end
end