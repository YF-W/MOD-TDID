function [tensor_graph] = apply_tensor_svd_regularization(Gi)
    num_views = length(Gi);
    [n, ~] = size(Gi{1});
    tensor_G = zeros(n, n, num_views);
    for v = 1:num_views
        tensor_G(:, :, v) = Gi{v}; 
    end
    [U, S, V] = t_svd(tensor_G);
    tensor_G_reconstructed = t_product(U, S, V);
    tensor_graph = cell(num_views, 1);
    for v = 1:num_views
        tensor_graph{v} = tensor_G_reconstructed(:, :, v);
    end
end
function [U, S, V] = t_svd(tensor_G)
    [n1, n2, n3] = size(tensor_G);
    tensor_G_fft = fft(tensor_G, [], 3);
    U = zeros(n1, n1, n3);
    S = zeros(n1, n2, n3);
    V = zeros(n2, n2, n3);
    for i = 1:n3
        [U(:,:,i), S(:,:,i), V(:,:,i)] = svd(tensor_G_fft(:,:,i), 'econ');
    end
    U = ifft(U, [], 3);
    S = ifft(S, [], 3);
    V = ifft(V, [], 3);
end

function tensor_G_reconstructed = t_product(U, S, V)
    [n1, ~, n3] = size(U);
    [~, n2, ~] = size(V);
    tensor_G_fft = zeros(n1, n2, n3);
    for i = 1:n3
        tensor_G_fft(:,:,i) = U(:,:,i) * S(:,:,i) * V(:,:,i)';
    end
    tensor_G_reconstructed = real(ifft(tensor_G_fft, [], 3));
end
