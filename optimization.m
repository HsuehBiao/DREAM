function D = optimization(L,X,Y,AnchorNum,beta,gamma,lambda)
rng(1202,'twister');
%Initialization
xi = -1;
% lambda = 1/(L+L);%clustering term
viewNum = length(X);%number of views
N = length(Y);%number of sample
alpha = ones(viewNum,1) / viewNum;
MaxIter = 100;       %
innerMax = 10;

rhob = 1e-4; max_rhob = 10e10; pho_rhob = 1.2; %惩罚系数

sel_sample = X{1}(:,randsample(N, AnchorNum),:); % m*n
% sel_sample = X{1}(:,1:AnchorNum);
[pcaW, ~] = eigs(cov(sel_sample'), L); % m*L
B = sign(pcaW'*X{1}); % L*n

n_cluster = numel(unique(Y));

U = cell(1,viewNum);


C = B(:,randsample(N, n_cluster));
HamDist = 0.5*(L - B'*C);
[~,ind] = min(HamDist,[],2);
D = sparse(ind,1:N,1,n_cluster,N,N);
D = full(D);
CG = C*D;

% 初始化误差记录数组
hash_errors = [];
mat_errors = [];

% Dynamically generate XXT and XX
XXT = cell(1,viewNum);
XX = [];
for view = 1:viewNum
    XXT{view} = X{view}*X{view}'; %m*m x*x'
    XX = [XX, X{view}'];%n*(m*v)   
end

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
    [tensor_g, ~] = solve_G(tensor_u + 1/rhob*tensor_w,rhob,sX,1e-2);
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

    if err < 2 && iter > 1
        break;
    end
end

% 绘制收敛曲线
figure;
subplot(2,1,1);
plot(1:MaxIter, hash_errors, '-o', 'LineWidth', 2);
title('Hash Encoding Error vs Iteration');
xlabel('Iteration'); ylabel('Hash Error');
grid on;

subplot(2,1,2);
plot(1:MaxIter, mat_errors, '-s', 'LineWidth', 2);
title('Matrix Matching Error vs Iteration');
xlabel('Iteration'); ylabel('Matrix Error');
grid on;