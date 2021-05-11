function [AntecedentsB, WB]=fis2num(fis)
if nargin < 1
    rng(0)
    warning off all
    load('./fis.mat','fis')
end
NumInput=length(fis.Inputs);
NumMF=length(fis.Inputs(1, 1).MembershipFunctions);
AntecedentsB=nan(NumMF,NumInput,2);
WB=nan(NumMF,length(fis.Outputs(1).MembershipFunctions(1).Parameters));
for i=1:NumInput
    for j=1:NumMF
        AntecedentsB(j,i,:) = fis.Inputs(i).MembershipFunctions(j).Parameters;
    end
end
for j=1:NumMF
    WB(j,[2:end 1]) = fis.Outputs(1).MembershipFunctions(j).Parameters;
end
% data=load('./results.mat','AntecedentsB', 'WB');
% sum(abs(data.AntecedentsB-AntecedentsB),'all')
% sum(abs(data.WB-WB),'all')
end