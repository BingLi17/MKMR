% all other predictor steps
dsigma = min(sigmas(end)-sigmas(end-1)+0.001,maxdsigma);
compute_newlambda2
neval = neval + 1;
if isinf(lambda2) || isnan(lambda2) || (lambda2>EPS2)
    while isinf(lambda2) || isnan(lambda2) || (lambda2>EPS2)
        dsigma = dsigma/sqrt(2);
        compute_newlambda2;
        neval = neval + 1;
    end
    
else
    while (lambda2<=EPS2) & (dsigma <= maxdsigma)
        dsigma = dsigma*sqrt(2);
        compute_newlambda2;
        neval = neval + 1;
        if ( sigma + dsigma > maxsigma ) & ( lambda2 <= EPS2) ,
            % get beyond maximal value - stops
            dsigma = dsigma*sqrt(2);
            break; 
        end
    end
    dsigma = dsigma/sqrt(2);
    if boosted
        %     switch predictor_type
        %         case 0, newalpha = alpha + dsigma * (alpha_old-alpha)/(sigma_old-sigma);
        %         case 1, newalpha = alpha + dsigma * da_dsigma + .5 * dsigma^2 * (da_dsigma_old-da_dsigma)/(sigma_old-sigma);
        %         case 2, newalpha = alpha + dsigma * da_dsigma + dsigma^2 * .5 * d2a_dsigma2 + ...
        %                 1/6 * dsigma^3 * (d2a_dsigma2_old-d2a_dsigma2)/(sigma_old-sigma);
        %     end
        
        
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
end
    newlambda = exp(-sigma-dsigma);