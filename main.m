% データ準備
[X,Y] = iris_dataset;
X = X';
X = X(:,1:2); % 可視化のため2次元に絞る
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

% --- 実験1: 学習率による学習曲線の違い ---
figure('Name', 'Learning Curve Comparison');
lrs = [0.5, 0.1, 0.05];
colors = ['r', 'b', 'g'];
hold on;
for i = 1:length(lrs)
    lr = lrs(i);
    % 2値分類（クラス0 vs 他）で実験
    y_binary = double(Y == 0);
    gbdt = GBDT(LogisticLoss(), 50, 1, 0, lr);
    gbdt.fit(X, y_binary);
    plot(gbdt.loss_history, 'Color', colors(i), 'LineWidth', 2, 'DisplayName', sprintf('LR=%.2f', lr));
end
xlabel('Iterations');
ylabel('Logistic Loss');
title('Learning Rate Comparison (Class 0 vs Rest)');
legend;
grid on;
hold off;

% --- 実験2: 正則化パラメータ(gamma)による決定境界の違い ---
gammas = [0, 1, 5];
n_estimators = 20;
regLambda = 1;
learningRate = 0.1;

figure('Name', 'Decision Boundary with different Gamma');
for i = 1:length(gammas)
    g = gammas(i);
    gbdt = GBDT_multiclass(LogisticLoss(), n_estimators, regLambda, g, learningRate);
    gbdt.fit(X, Y);
    
    % Plot
    subplot(1, 3, i);
    mesh = 100;
    xVec = linspace(min(X(:,1)) - 0.5, max(X(:,1)) + 0.5, mesh);
    yVec = linspace(min(X(:,2)) - 0.5, max(X(:,2)) + 0.5, mesh);
    [mx,my] = meshgrid(xVec,yVec);
    mX = [mx(:), my(:)];
    mz = gbdt.predict(mX);
    mz = reshape(mz,mesh,mesh);
    
    contourf(mx,my,mz);
    hold on;
    % クラスごとに色を変えてプロット
    scatter(X(Y==0,1), X(Y==0,2), 30, 'r', 'filled', 'o');
    scatter(X(Y==1,1), X(Y==1,2), 30, 'g', 'filled', '^');
    scatter(X(Y==2,1), X(Y==2,2), 30, 'b', 'filled', 's');
    title(sprintf('Gamma = %d', g));
    xlabel('Sepal Length');
    ylabel('Sepal Width');
    hold off;
end
