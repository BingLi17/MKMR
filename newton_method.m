function [alpha,lambda2,exit_status,k,nevals,exit_param] = newton_method(Ks,y,type,alpha,ds,lambda,mu,indices,display,tol,kmax);
if isempty(alpha), 
    switch type.type,
        case 'regression', alpha = zeros(Ks.n,1); 
        case 'logistic', alpha = -y/lambda*0.00001;
        case 'logistic_unbalanced', alpha = -y/lambda*0.00001;
    end
end
dalpha=2*tol;
df = 2*sqrt(tol)/10;
k=1;
go_on = 1;
alpha_NEWTON = 0.3 ;
beta_NEWTON  = 0.5 ;
fxold = Inf;
alphaold = alpha;
nevals = 0;
lambda2old = 0;
while go_on & k<kmax & df > tol^2;
    % compute gradient and hessian
    [fx,gradient,descent] = objective_function(Ks,y,type,alpha,ds,lambda,mu,indices);
    
    if isinf(fx)
        if k==1, exit_status = 'infinite_lambda_0'; else exit_status = 'infinite_lambda'; end
        lambda2 = lambda2old;
        exit_param = lambda2old;
        alpha = alphaold;
        return;
    end
    
        
    if isempty(descent)
        if k==1, exit_status = 'no_inverse_hessian_0'; else exit_status = 'no_inverse_hessian'; end
        lambda2 = lambda2old;
        exit_param = lambda2old;
        alpha = alphaold;
        return;
    end
    
    df = fxold-fx;
    fxold = fx;
    
    
    lambda2 = - gradient' * descent ;
    
    if isnan(lambda2)
        if k==1, exit_status = 'nan_lambda_0'; else exit_status = 'nan_lambda'; end
        lambda2 = lambda2old;
        exit_param = lambda2old;
        alpha = alphaold;
        return;
    end
     
    lambda2old = lambda2;
    if display,
        fprintf('k=%d - lambda2=%e - f=%e - df=%e - dalpha=%e\n',k,lambda2,fx,df,norm(alphaold-alpha));    
    end
    alphaold=alpha;
    if lambda2 < 2 * tol 
        % end of newton iteration
        go_on = 0;
        exit_status = 'small_lambda2';
        exit_param = lambda2;
    else
        % backtracking line search
        tt=1;
        maxbacktrack=100;
        try
            alpha1 = alpha + tt * descent ;
        catch
        q=round(10*sum(clock));
            q
            save(sprintf('error_%d',q));
            error('wrong descent direction');    
        end
        ft= objective_function(Ks,y,type,alpha1,ds,lambda,mu,indices);
        nevals = nevals + 1;
        while ft > fx + alpha_NEWTON * tt * descent' * gradient & maxbacktrack>0
            tt = beta_NEWTON  * tt;
            alpha1 = alpha + tt * descent ;
            ft= objective_function(Ks,y,type,alpha1,ds,lambda,mu,indices);
            nevals = nevals + 1;
            maxbacktrack = maxbacktrack - 1;
        end
        if maxbacktrack>0
            alpha=alpha1;
        else
            go_on=0;
            exit_status = 'max_backtrack';
            exit_param = 0;
        end
        k=k+1;
    end
end
if k==kmax, exit_status = 'max_iterations'; exit_param= lambda2; end
if df<=tol^2, exit_status= 'df_small'; exit_param=df; end
