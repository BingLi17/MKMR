function [Ks_train,Ks_test,kernel_types] = create_path_kernel_matrices(datatrain,datatest,kernelparams)

% creates kernel matrices
% NB : all kernel matrices are normalized to unit trace

% TYPES OF KERNEL
LINEAR          = kernelparams.LINEAR ;
FULL_LINEAR     = kernelparams.FULL_LINEAR ;
CONSTANT        = kernelparams.CONSTANT ;
IDENTITY        = kernelparams.IDENTITY ;
POLYNOMIAL      = kernelparams.POLYNOMIAL ;
FULL_GAUSSIAN   = kernelparams.FULL_GAUSSIAN ;
SUBSET_GAUSSIAN = kernelparams.SUBSET_GAUSSIAN ;
NSIGMAS         = kernelparams.NSIGMAS ;
MAXSUBSETS      = kernelparams.MAXSUBSETS ;
POLY_MAXORDER   = kernelparams.POLY_MAXORDER ;
SIGMAGRID       = kernelparams.SIGMAGRID;
datatrain = datatrain';
datatest = datatest';

[d, ntrain] = size(datatrain);
[d, ntest]  = size(datatest);

sigmas  = 10.^ ( SIGMAGRID * (-NSIGMAS+1:2:NSIGMAS) );


% build Gaussian kernel matrices on all
k=1;
matlab7 =  str2num(version('-release')) >=14;
if FULL_GAUSSIAN
    for j=1:NSIGMAS
        if matlab7
            Ks_train{k} = exp( - single( sigmas(j)) * single( sqdist(datatrain, datatrain ) ) / d )/ntrain;
            Ks_test{k} = exp( - single( sigmas(j)) *  single( sqdist(datatrain, datatest ) ) / d )/ntrain;
        else
            Ks_train{k} = exp( - sigmas(j) *  sqdist(datatrain, datatrain )  / d )/ntrain;
            Ks_test{k} = exp( - sigmas(j) *   sqdist(datatrain, datatest )  / d )/ntrain;
        end
        kernel_types.type{k} = 'gaussian';
        kernel_types.params1{k} = 1:d;
        kernel_types.params2{k} = sigmas(j);
        k=k+1;
    end
end

% build Gaussian kernel matrices on singletons and subsets
if SUBSET_GAUSSIAN
    for i=1:d
        for j=1:NSIGMAS
            if matlab7
                Ks_train{k} = exp( - single( sigmas(j) ) * single(sqdist(datatrain(i,:), datatrain(i,:))) )/ntrain;
                Ks_test{k} = exp( - single(sigmas(j))* single(sqdist(datatrain(i,:), datatest(i,:)) ) )/ntrain;
            else
                Ks_train{k} = exp( - sigmas(j) * sqdist(datatrain(i,:), datatrain(i,:) ) )/ntrain;
                Ks_test{k} = exp( - sigmas(j) * sqdist(datatrain(i,:), datatest(i,:) ) )/ntrain;
            end
            kernel_types.type{k} = 'gaussian';
            kernel_types.params1{k} = i;
            kernel_types.params2{k} = sigmas(j);
            k=k+1;
        end
    end



    if MAXSUBSETS>1
        % build kernel matrices on pairs
        for i=2:d
            for i2=1:i-1
                for j=1:NSIGMAS
                    if matlab7
                        Ks_train{k} = exp( - sigmas(j) * sqdist(datatrain([i i2],:), datatrain([i i2],:) ) )/ntrain;
                        Ks_test{k} = exp( - sigmas(j) * sqdist(datatrain([i i2],:), datatest([i i2],:) ) )/ntrain;
                    else
                        Ks_train{k} = exp( - single(sigmas(j)) * single(sqdist(datatrain([i i2],:), datatrain([i i2],:) ) ))/ntrain;
                        Ks_test{k} = exp( - single(sigmas(j)) * single( sqdist(datatrain([i i2],:), datatest([i i2],:) ) ))/ntrain;
                    end
                    kernel_types.type{k} = 'gaussian';
                    kernel_types.params1{k} = [i i2];
                    kernel_types.params2{k} = sigmas(j);

                    k=k+1;
                end
            end
        end

    end
end

if LINEAR
    % add linear kernel matrices
    for i=1:d
        Ks_train{k} = datatrain(i,:)'*datatrain(i,:) ;
        Ks_test{k} = datatrain(i,:)'*datatest(i,:) ;
        if trace(Ks_train{k}) > 10^-30
            Ks_test{k} = Ks_test{k} / trace(Ks_train{k});
            Ks_train{k} = Ks_train{k} / trace(Ks_train{k});
        end
        if matlab7, Ks_test{k} = single(Ks_test{k}); Ks_train{k} = single(Ks_train{k}); end
        kernel_types.type{k} = 'linear';
        kernel_types.params1{k} = i;

        k=k+1;
    end
end

if FULL_LINEAR
    % add linear kernel matrices
    Ks_train{k} = datatrain'*datatrain;
    Ks_test{k} = datatrain'*datatest;
    Ks_test{k} = Ks_test{k} / trace(Ks_train{k});
    Ks_train{k} = Ks_train{k} / trace(Ks_train{k});
    if matlab7, Ks_test{k} = single(Ks_test{k}); Ks_train{k} = single(Ks_train{k}); end

    kernel_types.type{k} = 'linear';
    kernel_types.params1{k} = 1:d;
    k=k+1;
end



if IDENTITY
    % add identity matrix
    Ks_train{k} = eye( ntrain)/ntrain;
    Ks_test{k} = zeros(ntrain,ntest)/ntrain;
    if matlab7, Ks_test{k} = single(Ks_test{k}); Ks_train{k} = single(Ks_train{k}); end

    kernel_types.type{k} = 'identity';
    k = k + 1;
end

if CONSTANT
    % add constant matrix
    Ks_train{k} = ones( ntrain)/ntrain;
    Ks_test{k} = ones(ntrain,ntest)/ntrain;
    if matlab7, Ks_test{k} = single(Ks_test{k}); Ks_train{k} = single(Ks_train{k}); end

    kernel_types.type{k} = 'constant';
    k = k + 1;
end





if POLYNOMIAL
    for i=2:POLY_MAXORDER
        % add linear kernel matrices
        Ks_train{k} = datatrain'*datatrain / d ;
        maxK = max(abs(Ks_train{k}(:)));
        Ks_train{k} = ( 1 + .5 * Ks_train{k}/maxK ).^i;
        Ks_test{k} = datatrain'*datatest / d;
        Ks_test{k} = ( 1 + .5 * Ks_test{k}/maxK ).^i;
        Ks_test{k} = Ks_test{k} / trace(Ks_train{k});
        Ks_train{k} = Ks_train{k} / trace(Ks_train{k});
        if matlab7, Ks_test{k} = single(Ks_test{k}); Ks_train{k} = single(Ks_train{k}); end

        kernel_types.type{k} = 'polynomial';
        kernel_types.params1{k} = i;
        k=k+1;
    end
end


