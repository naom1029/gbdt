classdef LogisticLoss
    properties
    end
    methods
        function loss = loss(obj,y,y_pred)
            loss = y .* log(1 + exp(-y_pred)) + (1 - y) .* log(1 + exp(y_pred));
        end
        function dcost = grad(obj,y,y_pred)
            dcost = -y .* exp(-y_pred) ./ (1 + exp(-y_pred)) + (1 - y) .* exp(y_pred) ./ (1+exp(y_pred));
        end
        function ddcost = hess(obj,y,y_pred)
            prob = 1 ./ (1 + exp(-y_pred));
            ddcost = prob .* (1 - prob);
        end
    end
end