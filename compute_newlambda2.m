if boosted
    % assume the path is proportional to 1/lamba (may be useful for
    % regression)
        switch predictor_type
        case 0, newalpha = alpha ;
        case 1, newalpha = alpha + ( exp(dsigma) - 1 ) * da_dsigma;
        case 2, newalpha = alpha + ( exp(dsigma) - 1 ) * da_dsigma + ...
                +  .5 * ( exp(dsigma) - 1 )^2 * (  d2a_dsigma2 - da_dsigma );
    end

else
    switch predictor_type
        case 0, newalpha = alpha;
        case 1, newalpha = alpha+dsigma*da_dsigma;
        case 2, newalpha = alpha+dsigma*da_dsigma+ dsigma^2 * .5 * d2a_dsigma2;
    end
end
newlambda = exp(-sigma-dsigma);
if efficient_predictor
    % keep previous inverse hessian
    [f,grad] = objective_function(Ks,y,loss,newalpha,ds,newlambda,mu);
    if isinf(f), lambda2 = Inf;
    else lambda2 = norm(hessianinvchol0'*grad)^2; end
else
    [f,grad,descent,eta,lambda2] = objective_function(Ks,y,loss,newalpha,ds,newlambda,mu);
end