function path = follow_entire_path(Ks,y,loss,ds,path_params,Ks_test,ytest)

% FOLLOW THE ENTIRE PATH OF REGULARIZATION
%
% INPUT
% Ks : kernel matrices
% y : response
% loss : loss.type= 'regression' or 'logistic'
% ds : weights of the block 1-norm
% path_params : parameters of the path (defined below)
% Ks_test, ytest (optional) : testing data
%
% OUTPUT
% path.alphas                   dual variables
% path.etas                     weights of kernels
% path.sigmas                   log(lambda)
% path.netas                    number of "nonzero" weights
% path.ks                       number of corrector steps
% path.training_errors          
% path.testing_errors
% path.training_predictions     values of K * alpha
% path.testing_predictions      values of K * alpha


% INITIALIZATION OF CONSTANTS
mu                  = path_params.mu;                   % parameter of log-barrier
EPS1                = path_params.EPS1;                 % precision parameters of Newton steps (very small and fixed)
EPS2                = path_params.EPS2;                 % precision parameter of tube around path (adaptive)
predictor_type      = path_params.predictor_type;       % 1 : first order predictor, 2 : second order
efficient_predictor = path_params.efficient_predictor;  % 1 : efficient predictor steps, 0 : full steps
efficient_eta       = path_params.efficient_eta;        % real value : threshold eta for corrector steps, 0 : no threshold
% i.e., only consider the significant ones, but it is buggy because, a zero
% eta might become significant and negative (when a constraint is broken)
maxsigma            = path_params.maxsigma;             % maximum value of sigma = -log(lambda);
newton_iter1        = path_params.newton_iter1 ;         % number of iterations with no modification
newton_iter2        = path_params.newton_iter2  ;         % delta number of iterations under which EPS2 is divided/multiplied by 2
maxdsigma           = path_params.maxdsigma;            % maximal step
mindsigma           = path_params.mindsigma;            % minimal step
maxvalue_EPS2       = path_params.maxvalue_EPS2;        % maximum value of tolerance for predictor steps

problemtype     = problem_type(loss); % classification or regression
m               = Ks.m;

test = 0;       % test = 1 if performs testing online
if nargin>=6,
    if ~isempty(Ks_test), test = 1; end
end

 MIN_DSIGMA_OCURRED = 0;
 
% FIRST STEP
beta = target(y,loss);
Kbeta = kernel_operation(Ks,1,beta);
start = sqrt( beta' * Kbeta ) ./ ds ;
lambda = 1.3 * max(start);
sigma = -log(lambda);
alpha =  beta / lambda;

% selecte subsets of eta
etas_comp = get_etas(Ks,y,loss,alpha,ds,lambda,mu);
indices = find( etas_comp' .* ds.^2 / mu  > efficient_eta );
 
% first corrector step
[alpha,lambda2,exit_status,k]= newton_method(Ks,y,loss,alpha,ds,lambda,mu,indices,0,EPS1,500);
    [f,grad,hessianinvchol0,eta,lambda2,da_dsigma,d2a_dsigma2] = all_derivatives(Ks,y,loss,alpha,ds,lambda,mu);

   

    da_dsigma_old = da_dsigma;
d2a_dsigma2_old = d2a_dsigma2;
alpha_old = alpha;
sigma_old = sigma;

% store first step
alphas = alpha;
sigmas = sigma;
etas = eta;
netas = length(find( eta' .* ds.^2 > mu * 4 ) );
ks = k;
lambda2s = lambda2;
normgrads = norm(grad)^2;
nevals = 0;
EPS2s = EPS2;

ypred = - kernel_operation(Ks,1,alpha) * eta;
switch problemtype,
    case 'regression'
        training_error = sum( (ypred-y).^2 ) / length(y);

    case 'classification'
        ypreds = sign( ypred );
        training_error = length(find( abs(ypreds-y)> 0 )) / length(y);

    case 'classification_unbalanced'
        ypreds = sign( ypred );
        training_error = sum( y(:,2) .* ( abs(ypreds-y(:,1))> 0 ) ) / length(y);


end
training_errors = training_error;
training_predictions = ypred;
path.ytrain = y;
path.problemtype = problemtype;

if test
    path.ytest = ytest;
    yhat = - kernel_operation_test(Ks_test,1,alpha) * eta;
    ypred = yhat;
    switch problemtype,
        case 'regression'
            testing_error = sum( (ypred-ytest).^2 ) / length(ytest);
        case 'classification'
            ypreds = sign( ypred );
            testing_error = length(find( abs(ypreds-ytest)> 0 )) / length(ytest);
        case 'classification_unbalanced'
            ypreds = sign( ypred );
            testing_error = sum( ytest(:,2) .* ( abs(ypreds-ytest(:,1))> 0 ) ) / length(ytest);

    end
    testing_errors = testing_error;
    testing_predictions = ypred;
end


%fprintf('iter %d - sigma=%f - n_corr=%d - lambda2=%1.2e - EPS2=%1.2e - n_pred=%d - neta=%d\n',length(sigmas),sigmas(end),ks(end),lambda2s(end),EPS2,nevals(end),netas(end));
max_steps_no_progress = 6;

% PREDICTOR-CORRECTOR STEPS
% exits if upperbound on sigma is reached or Newton steps didnot converge
delta_sigma = 1;
try_all_predictors=0;
while sigma < maxsigma  & lambda2 < EPS2 & delta_sigma > 1e-6 & length(sigmas)<=2000

    % every 100 moves, reset to large tube around data
    if mod(length(sigmas),100)==1, EPS2                = path_params.EPS2;  end

    % choosing dsigma : predictor steps
    neval = 0;

    if length(sigmas)==1,
        boosted = 0;
        % first predictor step: try out several values
        dsigmas = 10.^[-5:.5:0];
        for idsigma=1:length(dsigmas);
            dsigma=dsigmas(idsigma);
            compute_newlambda2;
            neval = neval + 1;
        if isinf(lambda2) || isnan(lambda2) || (lambda2>EPS2),  break;  
            
            end;
        end
        dsigma = dsigmas(idsigma-1);
        switch predictor_type
            case 0, newalpha = alpha;
            case 1, newalpha = alpha+dsigma*da_dsigma;
            case 2, newalpha = alpha+dsigma*da_dsigma+ dsigma^2 * .5 * d2a_dsigma2;
        end
        newlambda = exp(-sigma-dsigma);

    else
        % for regression, if long stagnation with the same number of
        % kernels, the path is likely to be linear in 1/lambda
        if length(netas)>=10 & isequal(problemtype,'regression')
            if all(netas(end-9:end)==netas(end)), boosted = 1; fprintf(' boosted '); else boosted = 0; end
        else
            boosted = 0;
        end
        predictor_step;
    end

    predicted_type=predictor_type;
    if dsigma <= 1e-5
        switch predictor_type
            case 2,
                predictor_type = 1;
                predictor_step
                predicted_type = 1;
                if dsigma <= 1e-5
                    predictor_type = 0;
                    predictor_step
                    predicted_type=0;
                end
                predictor_type = 2;
            case 1,
                predictor_type = 0;
                predictor_step
                predictor_type = 1;
                predicted_type =0;
        end
    end

    if dsigma <= 1e-5,
        % not event the simpler predictor steps work, get out!
        break;
    end

    if try_all_predictors
        alpha_predicted = newalpha;
        dsigma_predicted = dsigma;
        predictor_type = 0;
        predictor_step;
        predictor_type = 1;
        if dsigma>dsigma_predicted,
            predicted_type=0;
            fprintf(' trivial is better ');
        else
            newalpha = alpha_predicted ;
            dsigma =  dsigma_predicted;
        end
    end


    go_on=1;
    counter = 1;

    % select etas
    etas_comp = get_etas(Ks,y,loss,newalpha,ds,lambda,mu);
    indices = find( etas_comp' .* ds.^2 / mu  > efficient_eta );

    % store for later use
    sigma0 = sigma;
    newalpha0 = newalpha;
    dsigma0=dsigma;
    alpha0=alpha;

    while go_on & counter <= max_steps_no_progress
        % try to perform Newton steps from predicted sigma
        % if does not converge, diminishes sigma and divides EPS2 by 8
        % if never converges after 4 times, exit
        counter = counter + 1;
        sigmapot = sigma + dsigma;
        try
            [alphapot,lambda2,exit_status,k,nevals_newton,exit_params]= newton_method(Ks,y,loss,newalpha,ds,exp(-sigmapot),mu,indices,0,EPS1,30);
        catch
            q=round(10*sum(clock));
            q
            save(sprintf('error_%d',q));
            error('error in newton method');
        end

        if strcmp(exit_status,'max_iterations') || lambda2 >  1e-5 || ...
                strcmp(exit_status,'infinite_lambda_0') || ...
                strcmp(exit_status,'no_inverse_hessian_0') || ...
                strcmp(exit_status,'nan_lambda_0')
            fprintf('Newton''s method takes too long!! dsigma=%e lambda2=%e\n',dsigma,lambda);
            dsigma = dsigma / 8;
            EPS2 = EPS2 / 8;
            sigmapot = sigma + dsigma;
            if boosted

                switch predicted_type
                    case 0, newalpha = alpha ;
                    case 1, newalpha = alpha + ( exp(dsigma) - 1 ) * da_dsigma;
                    case 2, newalpha = alpha + ( exp(dsigma) - 1 ) * da_dsigma + ...
                            +  .5 * ( exp(dsigma) - 1 )^2 * (  d2a_dsigma2 - da_dsigma );
                end

            else
                switch predicted_type
                    case 0, newalpha = alpha;
                    case 1, newalpha = alpha+dsigma*da_dsigma;
                    case 2, newalpha = alpha+dsigma*da_dsigma+ dsigma^2 * .5 * d2a_dsigma2;
                end
            end
        else go_on=0;
        end
    end
    if counter == max_steps_no_progress & go_on, break; end

    alpha_old = alpha;
    sigma_old = sigma;

    alpha = alphapot;
    delta_sigma = abs( sigma - sigmapot);
    sigma = sigmapot;
    lambda = exp(-sigma);


    % store step
    try
        da_dsigma_old = da_dsigma;
        d2a_dsigma2_old = d2a_dsigma2;
        [f,grad,hessianinvchol,eta,lambda2,da_dsigma,d2a_dsigma2] = all_derivatives(Ks,y,loss,alpha,ds,lambda,mu);


        if isinf(f)
            % if the value of f is infinite, then it means that the previous
            % corrector steps with less etas made one of the forgotten eta
            % active -> redo the Newton steps with all etas


            counter = 1;
            sigma = sigma0;
            dsigma = dsigma0;
            newalpha = newalpha0;
            alpha =alpha0;

            while go_on & counter <= max_steps_no_progress
                % try to perform Newton steps from predicted sigma
                % if does not converge, diminishes sigma and divides EPS2 by 8
                % if never converges after 4 times, exit
                counter = counter + 1;
                sigmapot = sigma + dsigma;
                try
                    [alphapot,lambda2,exit_status,k,nevals_newton,exit_params]= newton_method(Ks,y,loss,newalpha,ds,exp(-sigmapot),mu,1:m,0,EPS1,30);
                catch
                    q=round(10*sum(clock));
                    q
                    save(sprintf('error_%d',q));
                    error('error in newton method');
                end

                if strcmp(exit_status,'max_iterations') || lambda2 >  1e-5 || ...
                        strcmp(exit_status,'infinite_lambda_0') || ...
                        strcmp(exit_status,'no_inverse_hessian_0') || ...
                        strcmp(exit_status,'nan_lambda_0')
                    fprintf('Newton''s method takes too long!! dsigma=%e lambda2=%e\n',dsigma,lambda);
                    dsigma = dsigma / 8;
                    EPS2 = EPS2 / 8;
                    sigmapot = sigma + dsigma;
                    if boosted

                        switch predicted_type
                            case 0, newalpha = alpha ;
                            case 1, newalpha = alpha + ( exp(dsigma) - 1 ) * da_dsigma;
                            case 2, newalpha = alpha + ( exp(dsigma) - 1 ) * da_dsigma + ...
                                    +  .5 * ( exp(dsigma) - 1 )^2 * (  d2a_dsigma2 - da_dsigma );
                        end

                    else
                        switch predicted_type
                            case 0, newalpha = alpha;
                            case 1, newalpha = alpha+dsigma*da_dsigma;
                            case 2, newalpha = alpha+dsigma*da_dsigma+ dsigma^2 * .5 * d2a_dsigma2;
                        end
                    end
                else go_on=0;
                end
            end
            if counter == max_steps_no_progress & go_on, break; end

            alpha_old = alpha;
            sigma_old = sigma;

            alpha = alphapot;
            delta_sigma = abs( sigma - sigmapot);
            sigma = sigmapot;
            lambda = exp(-sigma);
            [f,grad,hessianinvchol,eta,lambda2,da_dsigma,d2a_dsigma2] = all_derivatives(Ks,y,loss,alpha,ds,lambda,mu);



        end

        % if the values of the derivatives with respect to the path are infinite or NAN, put them to zero
        if sum( isnan(da_dsigma) | isinf(da_dsigma) ),
            fprintf('derivative of path is undefined\n');
            fprintf('only doing zero order now (trivial predictors)\n');
            da_dsigma=zeros(size(da_dsigma));
            d2a_dsigma2=zeros(size(d2a_dsigma2));
            predictor_type = 0;
            try_all_predictors=0;

        end
        if sum( isnan(d2a_dsigma2) | isinf(d2a_dsigma2) ) & predictor_type ==2,
            fprintf('second derivative of path is undefined\n');
            fprintf('only doing first order now\n');
            d2a_dsigma2=zeros(size(d2a_dsigma2));
            predictor_type = 1;
            try_all_predictors=1;
        end


    catch
        q=round(10*sum(clock));
        q
        save(sprintf('error_%d',q));
        error('error in all derivatives - stack saved for examination');
    end
    lambda2s = [ lambda2s lambda2];
    normgrads = [ normgrads norm(grad)^2 ];
    alphas = [ alphas alpha];
    sigmas = [ sigmas sigma];
    etas = [ etas eta];
    netas = [ netas length(find( eta' .* ds.^2 > mu * 4 ) ) ];
    ks = [ ks k];
    nevals = [ nevals neval];

    % update EPS2
    if ~strcmp(exit_status,'df_small')
        EPS2=EPS2 * 2^ ( (newton_iter1 -k)/newton_iter2 );
    else
        EPS2=EPS2 * 2^ ( (newton_iter1 -k)/newton_iter2/2 );
    end
    if lambda2 >= EPS2/10, EPS2 = lambda2 * 10; end
    EPS2 = max(EPS2, 1e-5);
    EPS2 = min(EPS2, maxvalue_EPS2);
    EPS2s = [ EPS2s EPS2 ];
    ypred = - kernel_operation(Ks,1,alpha) * eta;
    switch problemtype,
        case 'regression'
            training_error = sum( (ypred-y).^2 ) / length(y);
        case 'classification'
            ypreds = sign( ypred );
            training_error = length(find( abs(ypreds-y)> 0 )) / length(y);
        case 'classification_unbalanced'
            ypreds = sign( ypred );
            training_error = sum( y(:,2) .* ( abs(ypreds-y(:,1))> 0 ) ) / length(y);
    end
    training_errors = [ training_errors training_error];
    training_predictions = [ training_predictions ypred];
    
    % if the training errors is zero, stop the path following soon
    if training_errors(end)<training_errors(1)*1e-6
       maxsigma = min(maxsigma,sigma+1); 
    end
    
    
    % perform testing if applicable
    if test
        yhat = - kernel_operation_test(Ks_test,1,alpha) * eta;
        ypred = yhat;
        switch problemtype,
            case 'regression'
                testing_error = sum( (ypred-ytest).^2 ) / length(ytest);
            case 'classification'
                ypreds = sign( ypred );
                testing_error = length(find( abs(ypreds-ytest)> 0 )) / length(ytest);
            case 'classification_unbalanced'
                ypreds = sign( ypred );
                testing_error = sum( ytest(:,2) .* ( abs(ypreds-ytest(:,1))> 0 ) ) / length(ytest);


        end
        testing_errors = [ testing_errors testing_error ];
        testing_predictions = [ testing_predictions ypred];

    end



    %fprintf('iter %d - sigma=%f - n_corr=%d - lambda2=%1.2e - EPS2=%1.2e - n_pred=%d - neta=%d - dsigma=%f - status=%s\n',length(sigmas),sigmas(end),ks(end),lambda2s(end),EPS2,nevals(end),netas(end),dsigma,exit_status);
    if dsigma < mindsigma,
        % after 6 times in a row with a small progress, exit
        MIN_DSIGMA_OCURRED = MIN_DSIGMA_OCURRED + 1;
        if MIN_DSIGMA_OCURRED == 6
            break;
        end
    else
        MIN_DSIGMA_OCURRED = 0;
    end
end

path.alphas=alphas;                                 
path.etas=etas;
path.sigmas=sigmas;
path.netas=netas;
path.ks=ks;
% path.nevals=nevals;
% path.normgrads = normgrads;
% path.lambda2s = lambda2s;
% path.EPS2s = EPS2s;
path.training_errors = training_errors;
path.training_predictions = training_predictions;
if test, path.testing_errors = testing_errors;
    path.testing_predictions = testing_predictions;
end

if 0,
    % debugging
    q=round(10*sum(clock));
    q
    save(sprintf('debug_%d',q));
    fprintf('results of path following saved');
end
