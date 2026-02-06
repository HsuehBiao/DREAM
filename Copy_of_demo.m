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
L = 16 ;            % Hashing code length
anchorNum = round(sqrt(length(Y)));
AnchorNum=max(L,anchorNum);

fprintf('Nonlinear Anchor Embedding...\n');
neighborNum = 10;
proximityOrder = 3;     % 高阶阶数

if ~exist('Anchor', 'var')
    Anchor = cell(1, viewNum); 
end

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

clear feaVec dist sigma dist Anchor it

%------------Initializing parameters--------------  
lambda=0.01;
gamma = 0.1;%自权重项的参数
beta=0.000000000001;%Bregman项的参数

disp('----------The proposed method (multi-view)----------');
tic
D = optimization(L,X,Y,AnchorNum,beta,gamma,lambda);
time=toc;
% profile on -memory

disp('----------Main Iteration Completed----------');
disp(['OUR-time：',num2str(time)]);
[~,pred_label] = max(D,[],1);
res_cluster = ClusteringMeasure(Y, pred_label);
fprintf('All view results: ACC = %.4f and NMI = %.4f, Purity = %.4f\n\n',res_cluster(1),res_cluster(2),res_cluster(3));