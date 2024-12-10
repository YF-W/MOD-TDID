function [inconsistency_scores] = MOD_TDID(X, k,alpha, gamma,maxiter)
    num_views = length(X);
    W = cell(num_views, 1); 
    E = cell(num_views, 1); 
    Gi = cell(num_views, 1); 
    order = 3; 
    H = cell(num_views, 1);
    V = cell(num_views, 1);
    for v = 1:num_views
        X{v} = zscore(X{v});
        distance_matrix = EuDist2(X{v}', X{v}');
        n = size(distance_matrix, 1);
        delta = sqrt(1 / (n * (n - 1)) * sum(distance_matrix(:).^2));
        W{v} = exp(-distance_matrix.^2 / (2 * delta^2));
        G = construct_knn_graph(W{v}, k);
        D = diag(sum(W{v}, 2)); 
        [L_opt]=construct_multilevel_spectral_L(D,W{v},G,order);
        
        E{v} = construct_anomaly_matrix(W{v},L_opt);
        W{v} = W{v} .* (1 - E{v});
        Gi{v} = construct_knn_graph(W{v}, k);
        H{v} = zeros(n);
        V{v} = Gi{v}; 
    end
    H_global = update_shared_embedding( V, Gi,E, gamma,0.0001,0.0000000001,maxiter);
    for v = 1:num_views
        V{v} = update_view_specific_embedding(V{v}, H_global, Gi{v},E{v}, gamma,0.00001,maxiter);
    end
    [tensor_graph] = apply_tensor_svd_regularization(Gi);
    S = fuse_graphs(H_global, Gi, tensor_graph, V, alpha);
    inconsistency_scores = compute_inconsistency_scores(S, Gi,E);
end

function E_final = construct_anomaly_matrix(W,L_opt)
    n = size(W, 1);
    E = zeros(n, n);
    local_density = sum(W, 2);
    [V, D_eig] = eig(L_opt);
    eigenvalues = diag(D_eig);
    eigenvalues(eigenvalues < 0) = 0;  
    for i = 1:n
        sim_set = W(i, :);
        mu_i = mean(sim_set);
        delta_i = std(sim_set);
        epsilon_i = (2 * mu_i - delta_i) / 2;
        for j = 1:n
            if i ~= j
                density_diff = abs(local_density(i) - local_density(j));
                eigenvalue_diff = abs(eigenvalues(i) - eigenvalues(j));
                
                if density_diff >= epsilon_i || eigenvalue_diff >= epsilon_i
                    E(i, j) =  0.000001 * (density_diff + eigenvalue_diff);
                end
            end
        end
    end
   E_final = (E + E') / 2 .* (V .* V);
end

function Gi = construct_knn_graph(W, k)
    n = size(W, 1);
    Gi = zeros(n, n);
    for i = 1:n
        [~, idx] = sort(W(i, :), 'descend');
        neighbors = idx(2:k+1);
        Gi(i, neighbors) = W(i, neighbors);
    end
    Gi = (Gi + Gi') / 2;
end

function H_global = update_shared_embedding(V, Gi, E, gamma, alpha, tol, max_iter)
    num_views = length(V);
    n = size(V{1}, 1);  
    H_global = randn(n);  
    prev_H_global = H_global;
    for iter = 1:max_iter
        HA = 0;
        HC = 0;
        for v = 1:num_views
            HA = HA + ((1 - E{v}) .* (V{v}' * V{v})) + 2 * gamma * (H_global' * H_global);
            HC = HC + ((1 - E{v}) .* ((V{v} - H_global)' * (Gi{v} - V{v})));
            for u = 1:num_views
                if u ~= v
                    HC = HC + alpha * (E{v}' * E{u});
                end
            end
        end
        H_global = sylvester(HA, HA', -HC);
        if norm(H_global - prev_H_global, 'fro') < tol
            break;
        end
        prev_H_global = H_global;
    end
    if iter == max_iter
    end
end


function V_new = update_view_specific_embedding(V_init, H_global, Gi, E, gamma, tol, max_iter)
    V_new = V_init;
    prev_V = V_new;
    n = size(V_init, 1);
    for iter = 1:max_iter
        enhanced_anomaly = E .* eye(n);  
        tVA = 2 * gamma * (H_global' * H_global) + enhanced_anomaly;
        tVB = eye(n) - Gi;
        V_new = sylvester(tVA, tVB, -(Gi - H_global));
        if norm(V_new - prev_V, 'fro') < tol
            break;
        end
        prev_V = V_new;
    end
    if iter == max_iter
    end
end





function S = fuse_graphs(H_global, Gi, tensor_graph, V, alpha)
    S = zeros(size(H_global));  
    num_views = length(Gi); 
    gamma_weights = zeros(1, num_views);
    for v = 1:num_views
        fro_norm = norm(H_global - Gi{v}, 'fro')^2; 
        anomaly_penalty = sum(V{v}(:)); 
        gamma_weights(v) = 1 / (2 * sqrt(fro_norm + anomaly_penalty)); 
        S = S + gamma_weights(v) * (Gi{v} .* (1 - alpha * tensor_graph{v}));
    end
    S = alpha * S + (1 - alpha) * H_global;
end


function inconsistency_scores = compute_inconsistency_scores(S, Gi, E)
    num_views = length(Gi);
    num_samples = size(Gi{1}, 1);
    inconsistency_scores = zeros(1, num_samples);
    for i = 1:num_samples
        similarities = zeros(1, num_views);
        gamma = zeros(1, num_views); 
        for v = 1:num_views
            fro_norm = norm(S(i, :) - Gi{v}(i, :), 'fro')^2; 
            anomaly_penalty = sum(E{v}(i, :));  
            gamma(v) = 1 / (2 * sqrt(fro_norm + anomaly_penalty));
            Gi{v}(i, :) = gamma(v) * Gi{v}(i, :);
            similarities(v) = (1 - adaptive_cosine_similarity(S(i, :), Gi{v}(i, :)));
        end
        inconsistency_scores(i) = mean(similarities);
    end
end

function similarity = adaptive_cosine_similarity(vec1, vec2)
    similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2));
end


