clear all
load('input_from_r.mat')

seed = 129;          % seed of random number generator
rng(seed, 'twister'); 

% simulation parameters
DS_POWER = 0;      
loss.type               = 'logistic';
nsplit = 500;       % number of splits
n = size(y, 1);          % number of data points
nfold = 5;               % number of folds

% path parameters
path_params.mu                  = 1e-3;      % parameter of log-barrier
path_params.EPS1                = 1e-10;     % precision parameters of Newton steps (very small and fixed)
path_params.EPS2                = 1e-2;      % precision parameter of tube around path (adaptive)
path_params.predictor_type      = 2;         % 1 : first order predictor, 2 : second order
path_params.efficient_predictor = 0;         % 1 : efficient predictor steps, 0 : full steps
path_params.efficient_eta       = 1;         % real value : threshold eta for corrector steps, 0 : no threshold
path_params.maxsigma            = 20;        % maximum value of sigma = -log(lambda);
path_params.newton_iter1        = 10;        % number of iterations with no modification
path_params.newton_iter2        = 6;         % delta number of iterations under which EPS2 is divided/multiplied by 2
path_params.maxdsigma           = 1;         % maximal step
path_params.mindsigma           = .004;      % minimal step
path_params.maxvalue_EPS2       = 1e-2;      % maximum value of tolerance for predictor steps

% record aucs
aucr = zeros(nsplit, 2);

for isplit = 1:nsplit
    disp(isplit);
    % split train and test
    fold = cvpartition(y','KFold', 2);
    ind_train = find(test(fold, 1)); ind_test = find(test(fold, 2));
    y_train = y(ind_train); y_test = y(ind_test);
   for j = 1:2
        switch j  
            % 3 kernels
            case 1, Ds = {D_BC, D_weighted, D_unweighted};
            % 7 kernels
            case 2, Ds = {D_BC, D_weighted, D_unweighted, D_0, D_0_25, D_0_5, D_0_75};
        end     
        m = length(Ds) + 1; 
        % cross validation
        Ks_train = cell(1, m);
        Ks_val = cell(1, m);
        crossv = cvpartition(y_train(:, 1)','KFold', 5);
        for k = 1:nfold
            % train & test
            for l = 1:(m-1)
                Ks_train{l} = D2K_Semi(Ds{l}(ind_train, ind_train), training(crossv, k));
                tmp =  D2K(Ds{l}(ind_train, ind_train));
                Ks_val{l} = tmp(training(crossv, k), test(crossv, k)); 
            end 
     
            Ks_train{l+1} = eye(length(ind_train(training(crossv, k))));
            Ks_val{l+1} = zeros(length(ind_train(training(crossv, k))), length(ind_train(test(crossv, k))));
            
            for l = 1:m
                trace_normalizer = trace(Ks_train{l});
                Ks_train{l} = Ks_train{l} / trace_normalizer;
                Ks_val{l} = Ks_val{l} / trace_normalizer;
            end
            ytrain = y_train(training(crossv, k), :);
            yval = y_train(test(crossv, k), :);
            % build kernels
            Kse_train = build_efficient_Ks(Ks_train, 1);
            Kse_val = build_efficient_Ks_test(Ks_val, 0);  
            ds = compute_ds(Ks_train, DS_POWER); 
            % create path
            path = follow_entire_path(Kse_train, ytrain, loss, ds, path_params, Kse_val, yval);
            paths{k} = path; 
        end
        % Note that the optimal thing to do to obtain an error
        % for a value of lambda which is not already samples is to
        % take alpha as a linear interpolation of the closest sampled points,
        % and then compute the prediction and the error.
        % In what follows, we simply linearly interpolate the errors.
      % In what follows, we simply linearly interpolate the errors.
        minsigma = 0;
        maxsigma = 16;
        for k = 1:nfold
            maxsigma = min(maxsigma, max(paths{k}.sigmas));
            minsigma = min(minsigma, min(paths{k}.sigmas));
        end

        nsigmas = 100;
        sigmas = minsigma + (0:(nsigmas - 1)) * (maxsigma - minsigma) / (nsigmas - 1);
        testing_errors = zeros(nfold, nsigmas);
        for k = 1:nfold
            path = paths{k};
            for isigma = 1:nsigmas
                sigma = sigmas(isigma);
                if sigma <= min(path.sigmas)
                    testing_errors(k, isigma) = path.testing_errors(1);
                elseif sigma >= max(path.sigmas)
                    testing_errors(k, isigma) = path.testing_errors(end);
                else
                    indm = min(find(path.sigmas>= sigma));
                    sigma1 = path.sigmas(indm - 1);
                    sigma2 = path.sigmas(indm);
                    te1 = path.testing_errors(indm - 1);
                    te2 = path.testing_errors(indm);
                    testing_errors(k, isigma) = te1 + (sigma - sigma1)/(sigma2 - sigma1)*(te2 - te1);
                end
            end
        end
        err = sum(testing_errors, 1);
        [minerror, ind] = min(err);
        msigma = sigmas(ind);      

       % test
        Ks_test = cell(1, m);
        for l=1:(m-1)
            Ks_train{l} = D2K_Semi(Ds{l}, ind_train);
            tmp = D2K(Ds{l});
            Ks_test{l} = tmp(ind_train, ind_test); 
        end
    
         Ks_train{l+1} = eye(length(ind_train));
         Ks_test{l+1} = zeros(length(ind_train), length(ind_test));
           
        for k=1:m
            trace_normalizer = trace(Ks_train{k});
            Ks_train{k} = Ks_train{k} / trace_normalizer;
            Ks_test{k} = Ks_test{k} / trace_normalizer;
        end
        
        Kse_train = build_efficient_Ks(Ks_train, 1);
        Kse_test = build_efficient_Ks_test(Ks_test, 0);
        ds = compute_ds(Ks_train, DS_POWER);
        
        path = follow_entire_path(Kse_train, y_train, loss, ds, path_params, Kse_test, y_test);
        [mpath, indp] = min(abs(msigma - path.sigmas));
        pred = path.testing_predictions(:, indp);
        % etas: the kernel weights
        % etas{i, j} = path.etas(:, indp);  
        [x, z, t, aucr(isplit, j)] = perfcurve(y_test, pred, 1);
   end
end

mean(aucr, 1)
quantile(aucr, [0.025, 0.975], 1)