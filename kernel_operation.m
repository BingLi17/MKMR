function output = kernel_operation(Ks,type,param,indices);
if nargin<4 indices = 1:Ks.m; end
matlab7 =  str2num(version('-release')) >=14;

switch type,
    case 1, % multiply all matrices by same vector
        output = zeros(length(param),length(indices));
        switch Ks.efficient_type
            case 0,
                for j=1:length(indices), output(:,j) = Ks.data{indices(j)} * param; end
            case 1,
                if matlab7
                                        for j=1:length(indices), output(:,j) = vectorize_product_single(Ks.data(:,indices(j)),param); end
                else
                                        for j=1:length(indices), output(:,j) = vectorize_product(Ks.data(:,indices(j)),param); end
                end
                %                 output = vectorize_product_all(Ks.data,param);
        end
    case 2, % linear combination of matrices
        output = zeros(Ks.n);
        switch Ks.efficient_type
            case 0,
                for j=1:lengh(indices), output = output +  Ks.data{indices(j)} * param(j); end
            case 1,
                if ~matlab7
                    output = devectorize( Ks.data(:,indices) * single(param) );
                else
                    output = devectorize_single( Ks.data(:,indices) * single(param) );
                    output = double(output);
                end
        end
end
