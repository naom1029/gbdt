classdef Tree < handle
    properties
        root
        regLambda
        gamma
        score
    end
    methods
        function obj = Tree(regLambda,gamma)
            obj.regLambda = regLambda;
            obj.gamma = gamma;
        end
        function gain = CalcGain(obj,gl,hl,gr,hr)
            Gl = sum(gl);
            Hl = sum(hl);
            Gr = sum(gr);
            Hr = sum(hr);
            
            % 原著論文でいうところのLsplit
            % 分割した時の左右のノードの評価関数が分割前より小さくなっていること
            gain = (Gl^2 / (Hl + obj.regLambda) + Gr^2 / (Hr + obj.regLambda) ...
                   - (Gl + Gr)^2 / (Hl + Hr + obj.regLambda)) / 2 - obj.gamma;
        end
        function score = calcBestScore(obj,gj,hj)
            score =- sum(gj) ./ (sum(hj) + obj.regLambda);
        end
     
        % 閾値(分割点)を求める
        function [best,threshold] = split(obj,node,grad,hess)% 適切な分割点を探索
            X = node.X;
            bestGain = 0;
            best = [];
            threshold = [];
            
            for feature=node.features %特徴量ごとにループ
                if length(unique(X(:,feature))) <=1
                    continue;
                end
                XX = X(:,feature);
                [~, ix] = sortrows(XX); %ソートしたイデックスを返す。
                x = X(ix,feature);
                grad = grad(ix);
                hess = hess(ix);
                
                % 計算量削減のため累積和
                cgrad = cumsum(grad);
                chess = cumsum(hess);
                for i = 2:length(x)
                    if x(i) == x(i-1)
                        % 分割する意味がないのでスキップ
                        continue;
                    end
                    gl = cgrad(i - 1);%分割点までの和
                    hl = chess(i - 1);
                    gr = cgrad(end) - cgrad(i - 1); %grad(i-1) +...+grad(end)%分割点以降の和
                    hr = chess(end) - chess(i - 1); %hess(i-1) +...+hess(end)
                    gain = obj.CalcGain(gl,hl,gr,hr);
                    % gain(Lsplit)が負の場合は分割しない
                    if gain > bestGain
                        best = feature;
                        threshold = (x(i) + x(i - 1)) / 2;
                        bestGain = gain;
                    end
                end
            end
        end
        
        function [] = fit(obj,X,grad,hess) % 決定木を生成
            obj.root = Node(X,0);
            obj.grow(obj.root,grad,hess);
        end
        
        function [] = grow(obj,node,grad,hess)
            X = node.X;
            [bestFeature,bestThreshold] = obj.split(node,grad,hess);
            if isempty(bestFeature)
                node.is_leaf = true;
                node.score = obj.calcBestScore(grad,hess);
                return;
            end
            node.is_leaf = false;
            node.feature = bestFeature;
            node.threshold = bestThreshold;
            il = X(:,bestFeature) <= bestThreshold;
            ir = X(:,bestFeature) > bestThreshold;
            ndepth = node.depth + 1;
            node.left = Node(X(il,:),ndepth);
            node.right = Node(X(ir,:),ndepth);
            obj.grow(node.left,grad(il),hess(il));
            obj.grow(node.right,grad(ir),hess(ir));
        end
                
        function predictions = predict(obj, X)
            [rows, ~] = size(X);
            predictions = zeros(rows, 1);
            
            for i = 1:rows
                x = X(i,:);
                current = obj.root;
                while ~current.is_leaf
                    if x(current.feature) <= current.threshold
                        current = current.left;
                    else
                        current = current.right;
                    end
                end
                
                predictions(i) = current.score;
            end
        end
    end
end
