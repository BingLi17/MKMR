function etas = get_etas(Ks,y,loss,alpha,ds,lambda,mu)
m = Ks.m;

Kalpha = kernel_operation(Ks,1,alpha,1:m);
temp = ds.^2 - alpha' * Kalpha;
etas = ( mu ./ temp )';
