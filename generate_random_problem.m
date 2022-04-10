function [X,Y] = generate_random_problem(n,m,d,noise,seed,type,parameter_type);
% generate a random problem for linear classification or regression
%
% INPUT
% n : number of data points
% m : number of kernels
% d : maximum dimension of each block
% noise : noise
% seed : random seed
% type : 'classification' or 'regression'
%
% OUTPUT
% X,Y : data

rand('state',seed);
randn('state',seed);

% random dimension of blocks
ps=floor(rand(1,m)*d)+1;
psc=[ 0 cumsum(ps)]; psc(end)=[];
for j=1:m
    ind{j}=psc(j)+1:psc(j)+ps(j);
end
p=sum(ps);
X=randn(n,p);
for j=1:m
    X(:,ind{j}) = randn(n,length(ind{j}));
    Ks{j}=X(:,ind{j}) * X(:,ind{j})';
end

% makes some Wj closer to zero
W = randn(p,1);
for j=1:m
    W(ind{j})=W(ind{j})*rand;    
end

rp = randperm(j); 
Y = X * W + noise * randn(n,1);
switch type,
    case 'classification'
        Y = round( (sign(Y)+1)/2 ); % labels in 0,1
        Y = 2*Y-1;                  % lables in -1,1
    case 'classification_unbalanced'
        
        [a,b] = sort(Y);
        Y = sign(Y-a(round(parameter_type*n)));
        Y = round( ((Y)+1)/2 ); % labels in 0,1
        Y = 2*Y-1;                  % lables in -1,1
        
end



