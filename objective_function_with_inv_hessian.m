function [f,grad,hessianinvchol,etas,lambda2,Kalpha] = objective_function_with_inv_hessian(Ks,y,loss,alpha,ds,lambda,mu,indices)
m = Ks.m;
if nargin<8, 
% indices: keeps indices for which eta is likely not to be zero
indices = 1:m;
end

etas=zeros(length(indices),1);

f = 1/lambda * psi_loss(lambda * alpha, y ,loss);

Kalpha = kernel_operation(Ks,1,alpha,indices);
temp = ds(indices).^2 - alpha' * Kalpha;
etas = ( mu ./ temp )';
if any(etas <= 0), 
    f = Inf; grad=[]; hessian=[]; hessianinvchol=[]; lambda2=Inf;
    return;
end
f = f - mu/2 * sum( log( temp ) );

if nargout>1
    
    grad = psi1_loss( lambda * alpha, y ,loss) +  Kalpha * etas;
    if nargout>2
        A = lambda * diag(  psi2_loss(lambda * alpha, y ,loss) );
        A = A + kernel_operation(Ks,2,etas,indices);
    end
end

if ~isinf(f) & nargout>2
    if max(etas) > lambda * 1e8
        ind = find( etas > max(etas)*1e-6);
    else
        ind = [];    
    end
    ind2 = setdiff(1:length(indices),ind);
    A = A + Kalpha(:,ind2) * diag( ( 2/mu ) .* etas(ind2).^2 ) * Kalpha(:,ind2)';
    
    try
        R = chol(A);
        maxdiag = max(diag(R));
        mindiag = min(diag(R));
        if  mindiag/maxdiag < 1e-4
            % this is for better conditioning of the inverse
        invR =  repmat( 1./(diag(R)) ,1 , size(R,1)) .* R; 
        invR = inv( invR );
        invR = invR .* repmat( 1./(diag(R))'  , size(R,1),1);
        else
            invR = inv(R);
        end
        if any(isnan(invR(:))) || any(isinf(invR(:))), 'pb3', error('pb'); end 
    catch
        fprintf('Hessian too ill-conditioned\n');
        hessianinvchol=[];    
        lambda2 = Inf;
        return;
    end
    
    if isempty(ind)
        hessianinvchol = invR;
    else
        B = Kalpha(:,ind) * diag( sqrt( 2/mu ) .* etas(ind) );
        temp = invR' * B;
        [U,S,V] = svd(temp);    
        %     [UU,EE] = eig(temp'*temp);
        hessianinvchol = invR * U * diag( sqrt( 1./( 1+ diag(S*S') ) ) );
    end
    lambda2 = norm( hessianinvchol' * grad )^2;
else
    hessianinvchol=[];    
    lambda2 = Inf;
end

