

if NORMALIZE,
    x= x -repmat(mean(x,1),size(x,1),1);
    toremove = [];
    for j=1:size(x,2)
        normalizer = ( std(x(:,j)) + (max(x(:,j)) - min(x(:,j)))/16 );
        if normalizer > 0
            x(:,j) = x(:,j) /normalizer ;
        else toremove = [ toremove j ];
        end
    end
    x(:,toremove)=[];
end

if isequal(problem_type,'regression')
    y = y - mean(y);
    y = y / std(y);
    
end


% randomize order
rand('state', seed);
randn('state', seed);
ind    = randperm(size(x,1));
ntotal = min( ntotal, size(x,1) );
x      = x(ind(1:ntotal),:);
y      = y(ind(1:ntotal),:);


for isplit = 1:nsplits
    isplit
    ind    = randperm(ntotal);
    x      = x(ind,:);
    y      = y(ind,:);
    
    
    % separate training set and testing set
    ntrain    = round(ntotal*proptrain);
    ntest     = ntotal - ntrain;
    xtrain    = x(1:ntrain,:);
    xtest     = x((ntrain+1):end,:);
    ytrain    = y(1:ntrain,:);
    ytest     = y((ntrain+1):end,:);
    
    
    % kernels parameters;
    kernelparams.LINEAR = 1;            % include the linear kernel on all variables
    kernelparams.FULL_LINEAR =1 ;       % include linear kernels on each of the variables separately
    kernelparams.CONSTANT = 1;          % include the constant kernel
    kernelparams.IDENTITY = 1;           % include the identity matrix
    kernelparams.POLYNOMIAL = 1;         % include polynomial kernel
    kernelparams.FULL_GAUSSIAN = 1;      % include Gaussian kernels on all variables
    kernelparams.SUBSET_GAUSSIAN = 1;    % include Gaussian kernels on subsets of variables
    kernelparams.NSIGMAS = 7;           % number of width parameters for Gaussian kernels
    kernelparams.MAXSUBSETS = 1;        % maximal size of subsets for Gaussian kernels
    kernelparams.POLY_MAXORDER = 4;     % maximal order of polynomial kernel
    kernelparams.SIGMAGRID = .125;      % spacing between values of the log width for Gaussian kernels
    
    % compute kernel matrices and weights
    [Ks_train,Ks_test,kernel_types] = create_path_kernel_matrices(xtrain,xtest,kernelparams);
    m=length(Ks_train);                 % number of kernels
    
    ds = compute_ds(Ks_train,DS_POWER);
    
    % path following parameters
    switch problem_type,
        case 'regression', loss.type='regression';
        case 'classification', loss.type='logistic';
    end
    efficient_type=1;
    Kse_train = build_efficient_Ks(Ks_train,efficient_type);
    Kse_test = build_efficient_Ks_test(Ks_test,0);
    
    path_params.mu                  = 1e-3;      % parameter of log-barrier
    path_params.EPS1                = 1e-11;     % precision parameters of Newton steps (very small and fixed)
    path_params.EPS2                = 1e-2;      % precision parameter of tube around path (adaptive)
    path_params.predictor_type      = 2;         % 1 : first order predictor, 2 : second order
    path_params.efficient_predictor = 0;         % 1 : efficient predictor steps, 0 : full steps
    path_params.maxsigma            = maxsigma;        % maximum value of sigma = -log(lambda);
    path_params.newton_iter1        = 10;         % number of iterations with no modification
    path_params.newton_iter2        = 6;         % delta number of iterations under which EPS2 is divided/multiplied by 2
    
    
    path = follow_entire_path(Kse_train,ytrain,loss,ds,path_params,Kse_test,ytest);
    paths{isplit}=path;
end


% Now compute averages over all splits
%
% Note that the optimal thing to do to obtain an error
% for a value of lambda which is not already samples is to 
% take alpha as a linear interpolation of the closest sampled points,
% and then compute the prediction and the error.
% In what follows, we simply linearly interpolate the errors.

minsigma = 0;
maxsigma = maxsigma;
for isplit=1:nsplits,
    maxsigma = min( maxsigma, max(paths{isplit}.sigmas) );
    minsigma = min( minsigma, min(paths{isplit}.sigmas) );
end

nsigmas = 1000;
sigmas = minsigma + (0:(nsigmas-1)) * ( maxsigma-minsigma ) / (nsigmas - 1);
training_errors = zeros(nsplits,nsigmas);
testing_errors = zeros(nsplits,nsigmas);
netas = zeros(nsplits,nsigmas);

for isplit = 1:nsplits
    isplit
    path = paths{isplit};
    for isigma = 1:nsigmas
        sigma = sigmas(isigma);
        if sigma <= min(path.sigmas)
            training_errors(isplit,isigma) = path.training_errors(1);
            testing_errors(isplit,isigma) = path.testing_errors(1);
            netas(isplit,isigma) = path.netas(1);
        elseif sigma >= max(path.sigmas)
            training_errors(isplit,isigma) = path.training_errors(end);
            testing_errors(isplit,isigma) = path.testing_errors(end);
            netas(isplit,isigma) = path.netas(end);

        else
            k = min( find( path.sigmas>= sigma ) );
            sigma1 = path.sigmas(k-1);
            sigma2 = path.sigmas(k);
            tr1 = path.training_errors(k-1);
            tr2 = path.training_errors(k);
            te1 = path.testing_errors(k-1);
            te2 = path.testing_errors(k);
            n1 = path.netas(k-1);
            n2 = path.netas(k);
            training_errors(isplit,isigma) = ...
                tr1 + ( sigma - sigma1 ) / ( sigma2 - sigma1 ) * ( tr2 - tr1 );
            testing_errors(isplit,isigma) = ...
                te1 + ( sigma - sigma1 ) / ( sigma2 - sigma1 ) * ( te2 - te1 );
            netas(isplit,isigma) = ...
                n1 + ( sigma - sigma1 ) / ( sigma2 - sigma1 ) * ( n2 - n1 );
            
        end
    end
end

save(name,'netas','testing_errors','training_errors','paths','sigmas');
