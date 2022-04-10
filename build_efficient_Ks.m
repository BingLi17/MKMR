function Kse = build_efficient_Ks(Ks,efficient_type);
% build efficient representation of the kernel matrices
% efficient_type = 0 -> not efficient
%                  1 -> efficient
matlab7 =  str2num(version('-release')) >=14;

Kse.efficient_type = efficient_type ;
Kse.m = length(Ks);
Kse.n = size(Ks{1},1);
switch efficient_type
    case 0
        Kse.data = Ks;
    case 1
        m = length(Ks);
        n = size(Ks{1},1);
        if matlab7
            Kse.data = zeros( size(Ks{1},1),m,'single');
        else
            Kse.data = zeros(n*(n+1)/2,m);
        end
        for j=1:m

            if matlab7
                Kse.data(:,j) = symmetric_vectorize_single(Ks{j});
            else
                Kse.data(:,j) = symmetric_vectorize(Ks{j});
            end
        end
end
