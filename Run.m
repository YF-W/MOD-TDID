clear
load("ProteinFold_5_258_12view.mat");
k = 25;
alpha = 0.1;
max_iter = 5;
num_experiments = length(X);
mean_auc = zeros(1, num_experiments);
for exp_idx = 1:num_experiments
    X_current = X{exp_idx};
    num_views = length(X_current);
    min_dim = inf;
    for v = 1:num_views
        min_dim = min(min_dim, min(size(X_current{v})));
    end
    adjusted_k = min(k, min_dim);
    [inconsistency_scores] = MOD_TDID(X_current,25,0.4,0.00000001,5);
    norm_inconsistency_scores = (inconsistency_scores - min(inconsistency_scores)) / (max(inconsistency_scores) - min(inconsistency_scores));
    gnd = out_label{exp_idx};
    [~, ~, ~, mean_auc(exp_idx)] = perfcurve(gnd, norm_inconsistency_scores, 1);
end
disp('Mean AUC for each experiment:');
disp(mean_auc);


