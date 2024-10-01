[X,Y] = iris_dataset;
X = X';
X = X(:,1:2);
% Yをonehot-eoncofingから戻す
for i=1:150
    if Y(1,i) == 1
        Y(1,i) = 0;
    elseif  Y(2,i) == 1
        Y(1,i) = 1;
    elseif Y(3,i) == 1
        Y(1,i) = 2;
    end
end
Y(2:3,:) = [];
% Train & Test
n_estimators = 15;
regLambda = 1;
gamma = 1;
learningRate = 0.1;
gbdt = GBDT_multiclass(LogisticLoss(),n_estimators,regLambda, gamma, learningRate);
gbdt.fit(X,Y);

% Plot & Draw
mesh = 200;
xVec = linspace(min(X(:,1)) - 1, max(X(:,1)) + 1, mesh);
yVec = linspace(min(X(:,2)) - 1, max(X(:,2)) + 1, mesh);
[mx,my] = meshgrid(xVec,yVec);
mX = [mx(:), my(:)];
mz = gbdt.predict(mX);
mz = reshape(mz,mesh,mesh);


contourf(mx,my,mz);
hold on;
scatter(X(1:50,1),X(1:50,2));
scatter(X(51:100,1),X(51:100,2),"+");
scatter(X(101:150,1),X(101:150,2),"x");
