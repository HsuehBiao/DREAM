function mat_error = compute_mat_error(U, G, viewNum,alpha)
    % 计算矩阵误差（公式33）
    % 输入:
    %   U: 投影矩阵的cell数组 {U{1}, U{2}, ..., U{V}} (每个U{v}是 l×m)
    %   G: 低秩张量的cell数组 {G{1}, G{2}, ..., G{V}} (每个G{v}是 l×m)
    %   viewNum: 视图数量 V
    % 输出:
    %   mat_error: 矩阵误差标量
    
    total_error = 0;
    for v = 1:viewNum
        % 计算无穷范数（最大绝对值）
        total_error = total_error + alpha(v)*norm(U{v} - G{v}, 'inf');
    end
    mat_error = total_error / viewNum;  % 平均误差
end