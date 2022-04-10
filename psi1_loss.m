function psi1 = psi1_loss(alpha, y ,loss);
switch loss.type
    case 'regression', 
        psi1 = alpha + y;
    case 'logistic', 
        alpha_Y = alpha.*y;
        psi1 = y.*log((1+alpha_Y)./(-alpha_Y));
    case 'logistic_unbalanced', 
        rho = y(:,2);
        y = y(:,1);
        
        alpha_Y = alpha.*y;
        psi1 = rho .* y .*log((1+alpha_Y)./(-alpha_Y));
end
