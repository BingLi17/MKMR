%% A function to get a kernel matrix
function kernel = D2K(distmat) 
    n_all = size(distmat, 1);
    H_all = diag(ones(n_all,1))- 1/n_all; 
    k_all = -1/2 * H_all *(distmat.^2) * H_all;
    [v, d] = eig(k_all);
    d = d .* (d >= 0);
    kernel = v * d * v';