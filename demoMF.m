clc; clearvars; close all; rng(0);
nRepeats=8;
nFs=1:6; % number of features
nRs=2.^nFs; % number of rules
MF = 'Gaussian';
Powerball = 0.5;
DropRule = 0.5;
lr = 0.01;
l2 = 0.05;
nIt = 10%00;
Nbs = 64;
LN0={'Triangular','Trapezoidal','BellShaped','Gaussian'};
LN=cell(1,length(LN0)*length(nRs)+1);
LN(1)={'RR'};
for i=1:length(nRs)
    LN(2+(i-1)*length(LN0):1+i*length(LN0))=strcat(LN0, ['-nR' num2str(nRs(i))]);
end
nAlgs=length(LN);

datasets={'Pyrim';'Triazines';'Estate-costs';'Estate-sales';'Musk1';'VAM-arousal';'VAM-dominance';'VAM-valence';'MusicOrigin-lat';'MusicOrigin-long';'MusicOriginPlus-lat';'MusicOriginPlus-long';'IAPS-Arousal';'IAPS-Dominance';'IAPS-Valence';'Isolet';'Communities';'Puma32h';'TIC';'Ailerons';'Pole'};
datasets=datasets(3)

% Display results in parallel computing
dqWorker = parallel.pool.DataQueue; afterEach(dqWorker, @(data) fprintf('%d-%d ', data{1},data{2})); % print progress of parfor

[RMSEtrain,RMSEtest,RMSEtune]=deal(cellfun(@(u)nan(length(datasets),nAlgs,nIt),cell(nRepeats,1),'UniformOutput',false));
[InitTimes,times,BestP,Bestalpha,BestgammaP]=deal(cellfun(@(u)nan(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false));
BestmIter=cellfun(@(u)ones(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false);
thres=cellfun(@(u)inf(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false);
BestF=cellfun(@(u)nan(length(datasets),nAlgs,max(nRs)),cell(nRepeats,1),'UniformOutput',false);
for r = 1
    % delete(gcp('nocreate'))
    % parpool(nRepeats);
    % parfor r=1:nRepeats
    dataDisp=cell(1,2);    dataDisp{1}=r;
    for s=1:length(datasets)
        dataDisp{2} = s;   send(dqWorker,dataDisp); % Display progress in parfor
        
        temp=load(['./' datasets{s} '.mat']);
        XTrain=temp.XTrain;
        XTune=temp.XTune;
        XTest=temp.XTest;
        yTrain=temp.yTrain;
        yTune=temp.yTune;
        yTest=temp.yTest;
        [N,M]=size(XTrain);
        N1=size(XTune,1);
        %% normalize y
        yTune=(yTune-mean(yTrain))/std(yTrain);
        yTest=(yTest-mean(yTrain))/std(yTrain);
        yTrain=(yTrain-mean(yTrain))/std(yTrain);
        
        nFs0=nFs;
        nFs0(nFs>M)=[];
        nFs0(2.^nFs0>N)=[];
        
        %% 1. Ridge regression
        tic
        id=1;
        b = ridge(yTrain,XTrain,l2,0);
        RMSEtrain{r}(s,id,:) = sqrt(mean((yTrain-[ones(N,1) XTrain]*b).^2));
        RMSEtune{r}(s,id,:) = sqrt(mean((yTune-[ones(N1,1) XTune]*b).^2));
        RMSEtest{r}(s,id,:) = sqrt(mean((yTest-[ones(length(yTest),1) XTest]*b).^2));
        times{r}(s,id)=toc;
        
        for nF=nFs0
            
            nRule=2^nF;
            
            %% Triangular
            tic;
            id=id+1;
            [tmp,tmpt,~, ~, ~, ~, fB]=sugfis_mbgd(XTrain,yTrain,{XTune,XTest},{yTune,yTest},'MF','tri','DR','CDR','nF',nF,'Init','FCM','nMF',nRule,'Opt','AdaBelief','Powerball',Powerball,'DropRule',DropRule,'lr',lr,'l2',l2,'nIt',nIt,'Nbs',Nbs);
            if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                [thres{r}(s,id),BestmIter{r}(s,id)]=min(tmpt{1});
                BestF{r}(s,id,1:nRule)=fB;
                [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
            end
            times{r}(s,id)=toc;
            
            %% Trapezoidal
            tic;
            id=id+1;
            [tmp,tmpt,~, ~, ~, ~, fB]=sugfis_mbgd(XTrain,yTrain,{XTune,XTest},{yTune,yTest},'MF','trap','DR','CDR','nF',nF,'Init','FCM','nMF',nRule,'Opt','AdaBelief','Powerball',Powerball,'DropRule',DropRule,'lr',lr,'l2',l2,'nIt',nIt,'Nbs',Nbs);
            if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                [thres{r}(s,id),BestmIter{r}(s,id)]=min(tmpt{1});
                BestF{r}(s,id,1:nRule)=fB;
                [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
            end
            times{r}(s,id)=toc;
            
            %% BellShaped
            tic;
            id=id+1;
            [tmp,tmpt,~, ~, ~, ~, fB]=sugfis_mbgd(XTrain,yTrain,{XTune,XTest},{yTune,yTest},'MF','gbell','DR','CDR','nF',nF,'Init','FCM','nMF',nRule,'Opt','AdaBelief','Powerball',Powerball,'DropRule',DropRule,'lr',lr,'l2',l2,'nIt',nIt,'Nbs',Nbs);
            if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                [thres{r}(s,id),BestmIter{r}(s,id)]=min(tmpt{1});
                BestF{r}(s,id,1:nRule)=fB;
                [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
            end
            times{r}(s,id)=toc;
            
            %% Gaussian
            tic;
            id=id+1;
            [tmp,tmpt,~, ~, ~, ~, fB]=sugfis_mbgd(XTrain,yTrain,{XTune,XTest},{yTune,yTest},'MF',MF,'DR','CDR','nF',nF,'Init','FCM','nMF',nRule,'Opt','AdaBelief','Powerball',Powerball,'DropRule',DropRule,'lr',lr,'l2',l2,'nIt',nIt,'Nbs',Nbs);
            if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                [thres{r}(s,id),BestmIter{r}(s,id)]=min(tmpt{1});
                BestF{r}(s,id,1:nRule)=fB;
                [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
            end
            times{r}(s,id)=toc;
            
        end
    end
end
save('demoMF.mat','MF','Powerball','DropRule','lr','l2','nIt','Nbs','RMSEtrain','RMSEtune','RMSEtest','times','BestF','BestmIter','thres','datasets','nAlgs','LN','LN0','nRepeats','nFs','nRs');


%% Plot results
ids=1:length(LN);
tmp=nan(length(datasets),length(LN),nRepeats);
for s=1:length(datasets)
    for id=1:length(LN)
        tmp(s,id,:)=cell2mat(cellfun(@(u,m)squeeze(u(s,id,m(s,id))),RMSEtest,BestmIter,'UniformOutput',false));
    end
end
lineStyles={'k','k','b','b','g','g','r','r';'-','--','-','--','-','--','-','--'};
close all
figure;
set(gcf,'DefaulttextFontName','times new roman','DefaultaxesFontName','times new roman','defaultaxesfontsize',12);
idM=1:length(LN0);
f=mat2cell(permute(tmp(:,idM+(1:length(LN0):size(tmp,2)-length(LN0))',:),[3,2,1]),ones(1,nRepeats));
f=cellfun(@(x)permute(x,[2,3,1]),f,'UniformOutput',false);
RR=mat2cell(permute(tmp(:,1,:),[3,2,1]),ones(1,nRepeats));
RR=cellfun(@(x)permute(x,[2,3,1]),RR,'UniformOutput',false);
fR=cellfun(@(x,y)reshape(nanmean(x./y,2),length(nRs),[])',f,RR,'UniformOutput',false);
fR=cat(3,fR{:});
savgRMSE=nanmean(fR,3);
sstdRMSE=nanstd(fR,[],3);
for i=1:length(idM)
    errorbar(nRs, savgRMSE(i,:),sstdRMSE(i,:),'Color',lineStyles{1,i},'LineStyle',lineStyles{2,i},'linewidth',2);
    hold on;
end
set(gca,'XTick',nRs);
xlabel('$R$','interpreter','latex');
ylabel('Average normalized test RMSE');
set(gca,'yscale','log','xscale','log');
box on; axis tight;
legend(LN0(idM),'NumColumns',1,'Location','best','box','off');
