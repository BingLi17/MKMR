function psi2 = psi2_loss(alpha, y ,loss);
switch loss.type
    case 'regression', 
        psi2 = ones(size(y));
    case 'logistic', 
        alpha_Y = alpha.*y;
        psi2 = - 1./(alpha_Y.*(1+alpha_Y)); 
    case 'logistic_unbalanced', 
        rho = y(:,2);
        y = y(:,1);
        alpha_Y = alpha.*y;
        psi2 = - rho.* ( 1./(alpha_Y.*(1+alpha_Y)) ); 
        
end
