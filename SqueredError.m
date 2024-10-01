classdef SqueredError
    properties
    end
    methods
        function loss = loss(obj,y,y_pred)
            loss = (y-y_pred).^2;
        end
        function dcost = grad(obj,y,y_pred)
            dcost = -2 * (y - y_pred);
        end
        function ddcost = hess(obj,y,y_pred)
            ddcost = ones(size(y_pred)).*2;
        end
    end
end