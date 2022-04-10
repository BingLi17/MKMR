function beta = target(y,loss);
switch loss.type,
    case 'regression', beta = -y;
    case 'logistic',  beta = -y/2;
    case 'sparselogistic',  beta = -y/(1+exp(1-1/loss.lambda) );
    case 'logistic_unbalanced',  beta = -y(:,1)/2;
end
