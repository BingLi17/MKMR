function output = kernel_operation_test(Ks,type,param);

switch type,
    case 1, % multiply all matrices by same vector (by the left)
        output = zeros(Ks.ntest,Ks.m);
        switch Ks.efficient_type
            case 0,
                for j=1:Ks.m, output(:,j) = Ks.data{j}' * param; end
            case 1,
%                 for j=1:Ks.m, output(:,j) = vectorize_product(Ks.data(:,j),param); end
% %                 output = vectorize_product_all(Ks.data,param);
            end
        case 2, % linear combination of matrices
%             output = zeros(Ks.n);
%             switch Ks.efficient_type
%                 case 0,
%                     for j=1:Ks.m, output = output +  Ks.data{j} * param(j); end
%                 case 1,
%                     output = devectorize( Ks.data * param );
%                 end
        end
