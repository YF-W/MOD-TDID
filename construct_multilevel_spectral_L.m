function[L_opt]=construct_multilevel_spectral_L(D,W_v,G,order)
        L = D - W_v; 
        G_o = construct_high_order_adjacency(G, order);
        L_o = construct_high_order_laplacian(G_o);
        L_opt = optimize_L_newton(L, L_o); 
end

function L_o = construct_high_order_laplacian(G_o)  
    node_degrees = sum(G_o, 2);  
    D_o = diag(node_degrees); 
    D_o_inv = pinv(D_o);  
    D_o_inv_sqrt = D_o_inv^(1/2);  
    M = D_o_inv_sqrt * G_o * D_o_inv_sqrt;
    L_o = eye(size(G_o)) - 0.5 * (M + M');  
end
function L_opt = optimize_L_newton(L, L_o)
    [rows, cols, O] = size(L_o); 
        grad = 0; 
        for o = 1:O
            grad = grad + 2 * (L - L_o(:,:,o)); 
        end
        H = zeros(rows, cols); 
        for o = 1:O
            H = H + 2 * eye(rows); 
        end
        L = L - (H \ grad);
    L_opt = L;  
end

function G_o = construct_high_order_adjacency(G, order)
    G_o = G; 
    for o = 1:order
        G_o = G_o * G; 
    end
end