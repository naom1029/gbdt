classdef GBDT < handle
    properties
        objective    % 目的関数
        n_estimators % 決定木の数
        regLambda    % L2正則化項の係数
        gamma        % 葉の数を制限するための係数
        learningRate  % 学習率
        trees
        loss_history % 学習ごとの損失の履歴
    end
    methods
        function obj = GBDT(objective,n_estimators,regLambda,gamma,learningRate)
            obj.objective = objective;
            obj.n_estimators = n_estimators;
            obj.regLambda = regLambda;
            obj.gamma = gamma;
            obj.learningRate = learningRate;
            obj.loss_history = [];
        end
        function obj =  fit(obj,X,y)
            obj.trees = [];
            obj.loss_history = zeros(obj.n_estimators, 1);
            y_pred = zeros(size(y));
            for i=1:obj.n_estimators
                grad = obj.objective.grad(y,y_pred);
                hess = obj.objective.hess(y,y_pred);
                tree = Tree(obj.regLambda,obj.gamma);
                tree.fit(X,grad,hess);
                y_pred = y_pred + obj.learningRate*tree.predict(X)';
                obj.trees = [obj.trees;tree];
                
                % 損失を記録
                obj.loss_history(i) = sum(obj.objective.loss(y, y_pred));
            end
        end
        function y_pred = predict(obj,X)
           [rows, ~] = size(X);
            y_pred = zeros(length(X),1);
            for i=1:size(obj.trees)
                tree=obj.trees(i);
                y_pred = y_pred + obj.learningRate * tree.predict(X);
            end
        end
    end
end