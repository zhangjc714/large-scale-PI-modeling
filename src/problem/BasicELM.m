classdef BasicELM < handle
    % Extreme Learning Machine
    
    properties
        InputWeight;      % 输入层权重矩阵
        Bias;             % 隐层偏置向量
        OutputWeight;     % 输出层权重矩阵
        ActivationFunc;   % 激活函数句柄
        Lambda;           % 正则化系数
    end
    
    methods
        function obj = BasicELM(InputWeight, Bias, Activation, Lambda)
            if nargin < 4
                Lambda = 0;
            end
            obj.InputWeight = InputWeight;
            obj.Bias = Bias;
            obj.Lambda = Lambda;
            
            switch lower(Activation)
                case {'tanh', 'tansig'}
                    obj.ActivationFunc = @(x) tanh(x);
                case {'sig', 'sigmoid'}
                    obj.ActivationFunc = @(x) 1 ./ (1 + exp(-x));
                case {'sin', 'sine'}
                    obj.ActivationFunc = @(x) sin(x);
                case {'hardlim'}
                    obj.ActivationFunc = @(x) double(hardlim(x));
                case {'tribas'}
                    obj.ActivationFunc = @(x) tribas(x);
                case {'radbas'}
                    obj.ActivationFunc = @(x) radbas(x);
                otherwise
                    error('Unknown activation function: %s', Activation);
            end
        end
        
        function obj = train(obj, x, y)
            % 隐层输出矩阵
            H = obj.ActivationFunc(obj.InputWeight' * x' + obj.Bias');
            
            % 判断是否加正则化
            if obj.Lambda > 0
                % 加上 λI 的正则项
                HtH = H * H';
                I = eye(size(HtH));
                obj.OutputWeight = (HtH + obj.Lambda * I) \ (H * y);
            else
                % 原始 ELM
                obj.OutputWeight = pinv(H') * y;
            end
        end
        
        function y = predict(obj, x)
            H = obj.ActivationFunc(obj.InputWeight' * x' + obj.Bias');
            y = H' * obj.OutputWeight;
        end
    end
end
