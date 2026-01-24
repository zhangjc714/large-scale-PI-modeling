classdef DKL_LSMOEA < ALGORITHM
% DKL-LSMOEA: Dual Knowledge Learning-based Large-Scale Multi-Objective Evolutionary Algorithm

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            params = struct('t1', 20, 't2', 20, 'prob', 1.0, 'ratio', 0.1, 'maxNo', 100, ...
                    'reg', 1e-3, 'MaxIter', 1000, 'TolFun', 1e-4, 'scale', 1.1,  'numTree', 100);
            
            %% Initialization
            Population = Problem.Initialization();
            Dataset = struct('EDs', [], 'labels', []);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                %% Stage 1. Data collection
                gen = 1;
                while gen <= params.t1 && Algorithm.NotTerminated(Population)
                    current_pop = Population;
                    Population = Evolve1(Problem,Population);
                    next_pop = Population;
                    Dataset = CollectData(Dataset,current_pop,next_pop);
                    gen = gen + 1;
                end
                
                %% Stage 2. Dual knowledge learning
                Idx = LearnStructure(Problem,Dataset,params);  % Learn structural importance knowledge
                ED = LearnDirection(Problem,Dataset,Idx,params);  % Learn directional distribution knowledge
                Dataset = struct('EDs', [], 'labels', []);  % Clear dataset
                
                %% Stage 3. Knowledge-driven optimization
                gen = 1;
                while gen <= params.t2 && Algorithm.NotTerminated(Population)
                    Population = Evolve2(Problem,Idx,Population,ED,params);
                    gen = gen + 1;
                end
            end
        end
    end
end

function Population = Evolve1(Problem,Population)
    Offspring  = OperatorDE(Problem,Population,Population(randi(Problem.N,1,Problem.N)),Population(randi(Problem.N,1,Problem.N)));
    [Population,~,~] = EnvironmentalSelectionNSGAII([Population,Offspring],Problem.N);
end

function Dataset = CollectData(Dataset,current_pop,next_pop)
    for i = 1 : length(current_pop)
        p = current_pop(i);
        q = next_pop(i);
        Dataset.EDs = [Dataset.EDs; q.dec-p.decs];
        if all(q.objs<=p.objs) && any(q.objs<p.objs)
            label = 1;  % success
        else
            label = 0;  % fail
        end
        Dataset.labels = [Dataset.labels; label];
    end
end

function Population = Evolve2(Problem,Idx,Population,ED,params)
    % Knowledge-driven optimization
    Offspring = Population;
    for i = 1 : Problem.N
        P = randperm(Problem.N);
        if rand() < params.prob
            % Knowledge-driven reproduction operator
            OffDec = Population(i).decs + params.scale * ED(i,:);  % 只改变Idx指定的变量
            Offspring(i) = Problem.Evaluation(OffDec);
        else
            OffDec = Population(i).dec;
            NewDec = OperatorDE(Problem,Population(i).decs,Population(P(1)).decs,Population(P(2)).decs);
            OffDec(Idx) = NewDec(Idx);  % 只改变Idx指定的变量
            Offspring(i) = Problem.Evaluation(OffDec);
        end
    end
    [Population,~,~] = EnvironmentalSelectionNSGAII([Population,Offspring],Problem.N);
end

