%% A function to get a kernel matrix so that kernel values among training samples do not depend on any test samples
function kernel = D2K_Semi(distmat, TR_ind) 
%     A_mat = - 0.5 * distmat .^ 2;
%     A_mat_TR = A_mat(:, TR_ind);
%     rmean_A_mat_TR = mean(A_mat_TR, 2);
%     term23 = meshgrid(rmean_A_mat_TR) + meshgrid(rmean_A_mat_TR)'; 
%     term4 = mean(rmean_A_mat_TR(TR_ind));
%     kernel = A_mat - term23 + term4;
    dismat_sub = distmat(TR_ind, TR_ind);
    kernel = D2K(dismat_sub);