function psi3 = psi3_loss(alpha, y ,loss);
switch loss.type
    case 'regression', 
        psi3 = zeros(size(y));
    case 'logistic', 
        alpha_Y = alpha.*y;
        psi3 = - y ./ ( 1 + alpha_Y ) ./ ( 1 + alpha_Y ) + y ./ alpha_Y ./ alpha_Y;
    case 'logistic_unbalanced', 
        rho = y(:,2);
        y = y(:,1);
        alpha_Y = alpha.*y;
        psi3 = rho.* ( - y ./ ( 1 + alpha_Y ) ./ ( 1 + alpha_Y ) + y ./ alpha_Y ./ alpha_Y );
end
