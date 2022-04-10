function [f,grad,descent,etas,lambda2,Kalpha] = objective_function(Ks,y,loss,alpha,ds,lambda,mu,indices)
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
    f = Inf; grad=[]; hessian=[]; descent=[]; lambda2=Inf;
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
    if any(isnan(A(:))), keyboard; end
    try
        if isempty(ind)
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
                    if any(isnan(invR(:))) || any(isinf(invR(:))), 'pb1', error('pb'); end 

            descent = - invR * (invR' * grad);
            
        else
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
        if any(isnan(invR(:))) || any(isinf(invR(:))), 'pb2', error('pb'); end 

            B = Kalpha(:,ind) * diag( sqrt( 2/mu ) .* etas(ind) );
            temp = invR' * B;
            [U,S,V] = svd(temp);
            hessianinvchol = invR * U * diag( sqrt( 1./( 1+ diag(S*S') ) ) );
            descent = - hessianinvchol * ( hessianinvchol' * grad );
        end
    catch
        fprintf('Hessian too ill-conditioned\n');
        descent=[];
        lambda2 = Inf;
        return;
    end

    lambda2 = - descent' * grad;
else
    descent=[];
    lambda2 = Inf;
end

