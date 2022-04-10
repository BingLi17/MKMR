function ds = compute_ds(Ks_train,DS_POWER)
% compute the weights of the block 1-norm penalty
% assumes that kernels have unit trace
%
% INPUT:
% Ks_train : cell array of kernel matrices
% DS_POWER : parameter

ntrain = size(Ks_train{1},1);
for k=1:length(Ks_train)
    % check unit trace
    if isnan(Ks_train{k}), ds(k)=NaN;
    else
        if abs(trace(Ks_train{k})-1)>1e-4, k, abs(trace(Ks_train{k})-1), error('not unit trace'); end
        q=max(real(eig(Ks_train{k})),0);
        ds(k) = ( length(find(q>=.5 /ntrain ))).^( DS_POWER );
    end
end
ind = find(isnan(ds));
ds(ind) = min(ds);
