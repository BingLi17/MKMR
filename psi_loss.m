function psi = psi_loss(alpha, y ,loss);
switch loss.type
    case 'regression', 
        psi = (1/2)*alpha'*alpha + alpha'*y;
    case 'logistic', 
        alpha_Y = - alpha .*y;
        if any(alpha_Y <=0), psi = Inf; return; end
        if any(alpha_Y >= 1), psi = Inf; return; end
        psi = ( 1 - alpha_Y ) .* log( 1 - alpha_Y ) + alpha_Y .* log( alpha_Y );
        psi = sum(psi);
        
    case 'logistic_unbalanced', 
        rho = y(:,2);
        y = y(:,1);
        alpha_Y = - alpha .*y;
        if any(alpha_Y <=0), psi = Inf; return; end
        if any(alpha_Y >= 1), psi = Inf; return; end
        psi = ( 1 - alpha_Y ) .* log( 1 - alpha_Y ) + alpha_Y .* log( alpha_Y );
        psi = sum(rho.* psi);
    end
