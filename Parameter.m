clear;  
clear memory; 
clc;
addpath(genpath('./'));

rng(45,'twister');
dataname = "prokaryotic.mat";
load(dataname);
X=Data;
Y=Label;

 % Nonlinear anchor feature embedding
viewNum = size(X,2);
L = 16;            % Hashing code length
anchorNum = round(sqrt(length(Y)));
AnchorNum=max(L,anchorNum);

fprintf('Nonlinear Anchor Embedding...\n');
neighborNum = 10;
proximityOrder = 3;     % 高阶阶数

if ~exist('Anchor', 'var')
    Anchor = cell(1, viewNum); 
end

tic
for it = 1:viewNum

    X{it}=X{it}';
    if isempty(Anchor{it}) || size(Anchor{it},2) ~= AnchorNum
        rand_idx = randsample(size(X{it},2), AnchorNum); % 随机选样本索引
        % rand_idx=2:24;
        Anchor{it} = X{it}(:, rand_idx); % 从高阶矩阵抽取锚点 dimension * anchor_num
    end

    % 4. 高斯核映射（原有代码）
    dist = EuDist2(X{it}',Anchor{it}',0);%核心距离计算，n*m
    sigma = mean(min(dist,[],2).^0.5)*2;
    feaVec = exp(-dist/(2*sigma*sigma));%逐元素指数运算，n*m
    X{it} = bsxfun(@minus, feaVec', mean(feaVec',2));% Centered data，m*n

    kernel_features= X{it}'; %n*m
    % 1. 构建邻接矩阵 A^v
    Av = constructW_PKN(kernel_features, neighborNum);  % 转置以适配样本间相似度计算 m*m

    % 2. 生成高阶邻近矩阵 F(A^v)
    F_Av = zeros(size(Av)); % m*m
    current_A = Av;
    for j = 1:proximityOrder
        F_Av = F_Av + current_A;
        current_A = current_A * Av;
    end

    X_highorder = X{it}' * F_Av; % n*m
    X{it} = bsxfun(@minus, X_highorder', mean(X_highorder', 2));%m*n
end
time1 = toc;

clear feaVec dist sigma dist Anchor it

%------------grid research--------------

lambda_range = logspace(-5,3,9);%[1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1e+00, 1e+01, 1e+02, 1e+03]
gamma_range = logspace(-3,3,7);%[1e-03, 1e-02, 1e-01, 1e+00, 1e+01, 1e+02, 1e+03]
beta_range = 10.^(-12:-4);% [1.0e-12, 1.0e-11, 1.0e-10, 1.0e-09, 1.0e-08, 1.0e-07, 1.0e-06, 1.0e-05, 1.0e-04]

best_acc = 0;
best_nmi = 0;
best_purity = 0;
best_time = 0;
best_parameters = struct('lambda',[],'gamma',[],'beta',[]);
result = [];

for lambda = lambda_range
    for gamma = gamma_range
        for beta = beta_range
            fprintf('\n test_parameters:λ=%.6f,γ=%.6f,β=%.12f \n',lambda,gamma,beta);

            tic
            D = optimization(L,X,Y,AnchorNum,beta,gamma,lambda);
            time2=toc;

            total_time = time1+time2;

            [~,pred_label] = max(D,[],1);
            res_cluster = ClusteringMeasure(Y, pred_label);

            acc = res_cluster(1);
            nmi = res_cluster(2);
            purity = res_cluster(3);

            fprintf('\n result: acc=%.4f, nmi=%.4f, purity=%.4f time=%.2fs\n',acc,nmi,purity,total_time);

            % result = [result;[lambda, gamma, beta, acc, nmi, purity, total_time]];

            if acc > best_acc
                best_acc = acc;
                best_nmi = nmi;
                best_purity = purity;
                best_parameters.lambda = lambda;
                best_parameters.gamma = gamma;
                best_parameters.beta = beta;
                best_time = time2;
            end
        end
    end
end
disp(dataname);
fprintf('\n===== best_parameters =====\n');

fprintf('λ=%.6f, γ=%.6f, β=%.12f\n', ...
        best_parameters.lambda, best_parameters.gamma, best_parameters.beta);
fprintf('ACC=%.4f, NMI=%.4f, Purity=%.4f TIME=%.4f\n', best_acc, best_nmi, best_purity,best_time);