function loss = hash_objective(params, X, Y)
    % 解包所有参数
    L = params.L;
    gamma_val = params.gamma;
    omega_val = params.omega;
    beta_val = params.beta;
    
    % 固定参数设置
    viewNum = size(X, 2);
    xi = -1;  % 固定
    proximityOrder = 3;  % 固定
    neighborNum = 10;
    
    % 自动设置锚点数
    anchorNum = round(sqrt(length(Y)));
    AnchorNum = max(L, anchorNum);
    
    % ----------- 非线性锚点嵌入 ------------
    Anchor = cell(1, viewNum); 
    for it = 1:viewNum
        % 原始数据
        originalX = X{it};

        % 1. 构建邻接矩阵 A^v
        Av = constructW_PKN(originalX', neighborNum); 

        % 2. 生成高阶邻近矩阵 F(A^v)    
        F_Av = zeros(size(Av));
        current_A = Av;
        for j = 1:proximityOrder
            F_Av = F_Av + current_A;
            current_A = current_A * Av;
        end

        % 3. 替换原始特征
        X{it} = F_Av';  % 转置为 n_features × n_samples

        if isempty(Anchor{it}) || size(Anchor{it},2) ~= AnchorNum
            rand_idx = randsample(size(X{it},2), AnchorNum);
            Anchor{it} = X{it}(:, rand_idx);
        end

        % 4. 高斯核映射
        dist = EuDist2(X{it},Anchor{it}',0);
        sigma = mean(min(dist,[],2).^0.5)*2;
        feaVec = exp(-dist/(2*sigma*sigma));
        X{it} = bsxfun(@minus, feaVec', mean(feaVec',2));
    end
    clear feaVec dist sigma dist Anchor
    
    % ----------- 主优化循环 ------------
    % 使用参数设置
    lambda_val = 1/L;  % 根据L设置lambda
    rhob = 1e-4; max_rhob = 10e10; pho_rhob = 8; 
    
    % ... [这里插入您完整的主循环代码] ...
    % 从"MaxIter = 20;"到"res_cluster = ClusteringMeasure(...)"
    MaxIter = 20;       %
innerMax = 10;

rhob = 1e-4; max_rhob = 10e10; pho_rhob = 8; %惩罚系数
omega=1;%张量约束的系数R
% omega=0;
% lambda = 2e-5;   % 聚类项Hyper-para lambda 
lambda = 1/L;
xi = -1;%自权重项的系数
gamma = 1e-3;%自权重项的参数
beta=1e-6;%Bregman项的参数



N = size(X{1},2);%样本数量
rand('seed',1024);

sel_sample = X{1}(:,randsample(N, AnchorNum),:); 
[pcaW, ~] = eigs(cov(sel_sample'), L);   
B = sign(pcaW'*X{1});  

n_cluster = numel(unique(Y));
alpha = ones(viewNum,1) / viewNum; 
U = cell(1,viewNum);

rand('seed',500); 
C = B(:,randsample(N, n_cluster)); 
HamDist = 0.5*(L - B'*C);
[~,ind] = min(HamDist,[],2);
D = sparse(ind,1:N,1,n_cluster,N,N);
D = full(D);
CG = C*D;

% Dynamically generate XXT and XX
XXT = cell(1,viewNum);
XX = [];
for view = 1:viewNum
    XXT{view} = X{view}*X{view}';
    XX = [XX, X{view}'];
end

clear HamDist ind initInd n_randm pcaW sel_sample view

% Dynamically generate XXXT
block_size = size(XXT{1}, 1);
XXXT = zeros(block_size*viewNum, block_size*viewNum);
for i = 1:viewNum
    start_i = (i-1)*block_size + 1;
    end_i = i*block_size;
    XXXT(start_i:end_i, start_i:end_i) = XXT{i};
end

% Dynamically generate CC
intr = (viewNum-1).*eye(AnchorNum);
inter = eye(AnchorNum);
CC = zeros(AnchorNum*viewNum, AnchorNum*viewNum);
for i = 1:viewNum
    start_i = (i-1)*AnchorNum + 1;
    end_i = i*AnchorNum;
    CC(start_i:end_i, start_i:end_i) = intr;
    for j = 1:viewNum
        if i ~= j
            start_j = (j-1)*AnchorNum + 1;
            end_j = j*AnchorNum;
            CC(start_i:end_i, start_j:end_j) = -inter;
        end
    end
end

sX=[L,AnchorNum,viewNum];
for k=1:viewNum        
     TG{1}{k}=double(zeros(L,AnchorNum));% low-rank tensor    
     TW{1}{k}=double(zeros(L,AnchorNum)); % Lagrange multiplier for W        
end

intr=(viewNum-1).*eye(AnchorNum);%
inter=eye(AnchorNum);  
% CC=[intr,-inter,-inter,-inter,-inter;
%    -inter,intr,-inter,-inter,-inter;
%    -inter,-inter,intr,-inter,-inter;
%    -inter,-inter,-inter,intr,-inter;
%    -inter,-inter,-inter,-inter,intr;
%    ];

% XXXT=[XXT{1},zeros(size(XXT{1})),zeros(size(XXT{1})),zeros(size(XXT{1})),zeros(size(XXT{1}));
%        zeros(size(XXT{1})),XXT{2},zeros(size(XXT{1})),zeros(size(XXT{1})),zeros(size(XXT{1}));
%        zeros(size(XXT{1})),zeros(size(XXT{1})),XXT{3},zeros(size(XXT{1})),zeros(size(XXT{1}));
%        zeros(size(XXT{1})),zeros(size(XXT{1})),zeros(size(XXT{1})),XXT{4},zeros(size(XXT{1}));
%        zeros(size(XXT{1})),zeros(size(XXT{1})),zeros(size(XXT{1})),zeros(size(XXT{1})),XXT{5};
%        ];
clear inter intr

% 初始化误差记录数组
hash_errors = [];
mat_errors = [];

disp('----------The proposed method (multi-view)----------');
tic
% profile on -memory
for iter = 1:MaxIter

    fprintf('The %d-th iteration...\n',iter);
    %---------Update Ui--------------
    %% 
    UX = zeros(L,N);
    TWW{1} = cell2mat(TW{1}); % Convert cell array to matrix for computation
    TGG{1} = cell2mat(TG{1}); % Convert cell array to matrix for computation
    
    UU=(2*B*XX+rhob*TGG{1}-TWW{1})/(2*XXXT+rhob*eye(size(XX,2))+2*beta.*CC);%   

    %% %---------Update B--------------
    for v=1:viewNum
        U{v}=UU(:,1+(v-1)*AnchorNum:AnchorNum*v); 
        UX   = UX+alpha(v)*U{v}*X{v};
    end
    B = sign((UX+lambda*CG)/(1+lambda));
    B(B==0) = -1;
    %% %----------Update TG and TW ---------------
   tensor_u=[];
   tensor_w=[];
   tensor_u=[tensor_u(:);UU(:)];
   tensor_w=[tensor_w(:);TWW{1}(:)];            
   % [tensor_g, objV] = wshrinkObj(tensor_u + 1/rhob*tensor_w,omega/rhob,sX,0,3);
   [tensor_g, objV] = solve_G(tensor_u + 1/rhob*tensor_w,rhob/omega,sX,1e-2);
   tensor_g=reshape(tensor_g,sX);%

   for v=1:viewNum
       TG{1}{v} = tensor_g(:,:,v);
       TW{1}{v}=TW{1}{v}+rhob*(U{v}-TG{1}{v});
   end
%    fprintf('    norm_z %7.10f   \n ',norm_Z);
   rhob = min(rhob*pho_rhob, max_rhob);
    %----------------------------------
    %% 
    %---------Update C and G--------------
    for iterInner = 1:innerMax
        % For simplicity, directly using DPLM here
        C = sign(B*D'); 
        C(C==0) = 1;
        rho = .001; mu = .01; % Preferred for this dataset
        for iterIn = 1:3
            grad = -B*D' + rho*repmat(sum(C),L,1);
            C    = sign(C-1/mu*grad); C(C==0) = 1;
        end
        HamDist = 0.5*(L - B'*C); % Hamming distance referring to "Supervised Hashing with Kernels"
        [~,indx] = min(HamDist,[],2);
        D = sparse(indx,1:N,1,n_cluster,N,N);
    end
    CG = C*D;
    %% 
    %% ——————更新 alpha——————
    for j = 1:viewNum
        J_var{j} = norm(B - U{j}*X{j}, 'fro') ;
    end
   sum_gam = 0;
    for i = 1:viewNum
        alpha(i) = (- J_var{i} / gamma / xi)^(1 / (xi - 1));
        sum_gam = sum_gam + alpha(i);
    end

    %% 计算当前迭代的误差
    current_hash_error = compute_hash_error(B, U, X, viewNum, alpha);
    current_mat_error = compute_mat_error(U, TG{1}, viewNum, alpha);

    % 保存误差值
    hash_errors(iter) = current_hash_error;
    mat_errors(iter) = current_mat_error;
    err = max(hash_errors(iter),mat_errors(iter));

    if err < 1e-5 && iter > 1
        break;
    end


end
time2 = toc;
disp('----------Main Iteration Completed----------');
disp(['OUR-time：',num2str(time1+time2)]);
[~,pred_label] = max(D,[],1);
res_cluster = ClusteringMeasure(Y, pred_label);
    % 返回损失(1-ACC)
    loss = 1 - res_cluster(1);
    
    % 记录参数组合
    fprintf('L=%d, γ=%.1e, ω=%.1e, β=%.1e => Loss=%.4f\n', ...
            L, gamma_val, omega_val, beta_val, loss);
end