function Kse = build_efficient_Ks_test(Ks,efficient_type);
% build efficient representation of the kernel matrices for testing
% efficient_type = 0 -> not efficient
%                  1 -> efficient (currently not available)

matlab7 =  str2num(version('-release')) >=14;

Kse.efficient_type = efficient_type ;
Kse.m = length(Ks);
Kse.ntrain = size(Ks{1},1);
Kse.ntest = size(Ks{1},2);
switch efficient_type
    case 0
        Kse.data = Ks;    
    case 1
        %  not available
        Kse.data = Ks;    
end
