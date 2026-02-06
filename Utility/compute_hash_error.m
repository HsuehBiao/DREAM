function hash_error = compute_hash_error(B, U, X, viewNum,alpha)
    % 计算哈希误差（公式32）
    % 输入:
    %   B: 二进制编码矩阵 (l×N)
    %   U: 投影矩阵的cell数组 {U{1}, U{2}, ..., U{V}} (每个U{v}是 l×m)
    %   X: 映射后的特征cell数组 {X{1}, X{2}, ..., X{V}} (每个X{v}是 m×N，即φ(X^(v)))
    %   viewNum: 视图数量 V
    % 输出:
    %   hash_error: 哈希误差标量
    
    total_error = 0;
    for v = 1:viewNum
        % 计算 sgn(U^(v) * φ(X^(v)))
        B_pred = sign(U{v} * X{v});  % l×N
        B_pred(B_pred == 0) = -1;    % 处理零值（与B生成逻辑一致）
        
        % 累加各视图的Frobenius范数误差
        total_error = total_error + alpha(v)*norm(B - B_pred, 'fro');
    end
    hash_error = total_error / viewNum;  % 平均误差
end