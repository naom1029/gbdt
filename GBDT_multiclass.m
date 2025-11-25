classdef GBDT_multiclass < handle
    properties
        objective    % 目的関数
        n_estimators % 決定木の数
        regLambda    % L2正則化項の係数
        gamma        % 葉の数を制限するための係数
        learningRate  % 学習率
        models
    end
    methods
        function obj = GBDT_multiclass(objective,n_estimators,regLambda,gamma,learningRate)
            obj.objective = objective;
            obj.n_estimators = n_estimators;
            obj.regLambda = regLambda;
            obj.gamma = gamma;
            obj.learningRate = learningRate;
        end
        function obj =  fit(obj,X,y)
            n_classes = length(unique(y));
            obj.models = [];
            for k=0:n_classes - 1
                y_k = (y == k);
                y_k = double(y_k);
                model = GBDT(obj.objective, obj.n_estimators, obj.regLambda, obj.gamma, obj.learningRate);
                model.fit(X,y_k);
                obj.models =[obj.models;model];
            end
        end
        function ret = predict(obj,X)
            n_classes = length(obj.models);
            y_pred = zeros(size(X,1),n_classes);
            for k=1:n_classes
                y_pred(:,k) = obj.models(k).predict(X);
            end
            [~, ret] = max(y_pred, [], 2);
            ret = ret - 1;
        end
    end
end