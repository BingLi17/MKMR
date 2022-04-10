function [f,grad,hessianinvchol,eta,lambda2,da_dsigma,d2a_dsigma2] = all_derivatives(Ks,y,loss,alpha,ds,lambda,mu);
m=Ks.m;
[f,grad,hessianinvchol,eta,lambda2] = objective_function_with_inv_hessian(Ks,y,loss,alpha,ds,lambda,mu);


psi2 = psi2_loss(lambda * alpha, y ,loss);
psi3 = psi3_loss(lambda * alpha, y ,loss);
da_dlambda = alpha .* psi2;
da_dlambda = - hessianinvchol * ( hessianinvchol' * da_dlambda );
temp11 = hessianinvchol * ( hessianinvchol' * ( psi2 .* alpha ) ) ;
temp22 = kernel_operation(Ks,1,temp11);
temp3 = kernel_operation(Ks,1,alpha);
temp3 = hessianinvchol * ( hessianinvchol' * temp3 );

temp4 = (  2 * psi2 + 2 * lambda * psi3 .* alpha ) .* temp11 - alpha .* alpha .* psi3 ;
d2a_dlambda2 = hessianinvchol * ( hessianinvchol' * temp4 );

for j=1:m
    temp44 = sum( alpha.*temp22(:,j) );
    d2a_dlambda2 = d2a_dlambda2 - 2/mu*eta(j)^2 * temp44 * hessianinvchol * ( hessianinvchol' * temp22(:,j) );
    d2a_dlambda2 = d2a_dlambda2 - 2/mu*eta(j)^2 * temp44 * hessianinvchol * ( hessianinvchol' * temp22(:,j) );
    d2a_dlambda2 = d2a_dlambda2 - 8*eta(j)^3/mu/mu * temp44 * temp44 * temp3(:,j);
    d2a_dlambda2 = d2a_dlambda2 - 2/mu*eta(j)^2 * sum( temp11 .* temp22(:,j) ) * temp3(:,j);
end
d2a_dlambda2 = d2a_dlambda2 - lambda^2 * hessianinvchol * ( hessianinvchol' * ( psi3 .* temp11 .* temp11 ) );     


da_dsigma = da_dlambda * ( -lambda );
d2a_dsigma2 = d2a_dlambda2 * lambda * lambda +  da_dlambda * ( lambda );


