clc; clearvars; close all; rng(0);
nRepeats=8;
nFs=1:6; % number of features
nRs=2.^nFs; % number of rules
MF = 'Gaussian';
Powerball = 0.5; Powerballs = 0:.1:1.1;
DropRule = 0.5; DropRules = .1:.1:1;
lr = 0.01; lrs = 10.^(0:-0.5:-4);
l2 = 0.05; l2s = [0.001,0.005,0.01,0.05,0.1,0.5];
nIt = 1000;
Nbs = 64;
LN00={'CDR-FCM-RDpA'};
LN0=strcat(repmat(LN00,size(lrs)),'-lr',reshape(repmat(cellstr(string(log10(lrs))),length(LN00),1),1,[]));
LN0=[LN0 strcat(repmat(LN00,size(l2s)),'-l2',reshape(repmat(cellstr(string(l2s)),length(LN00),1),1,[]))];
LN0=[LN0 strcat(repmat(LN00,size(DropRules)),'-DropRule',reshape(repmat(cellstr(string(DropRules)),length(LN00),1),1,[]))];
LN0=[LN0 strcat(repmat(LN00,size(Powerballs)),'-Powerball',reshape(repmat(cellstr(string(Powerballs)),length(LN00),1),1,[]))];
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
[times,Bestlr,Bestl2,BestDropRule,BestPowerball]=deal(cellfun(@(u)nan(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false));
BestmIter=cellfun(@(u)ones(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false);
thres=cellfun(@(u)inf(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false);
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
        id=1;
        b = ridge(yTrain,XTrain,l2,0);
        RMSEtrain{r}(s,id,:) = sqrt(mean((yTrain-[ones(N,1) XTrain]*b).^2));
        RMSEtest{r}(s,id,:) = sqrt(mean((yTest-[ones(length(yTest),1) XTest]*b).^2));
        
        for nF=nFs0
            nRule=2^nF;
            
            for alpha=lrs
                tic;
                id=id+1;
                [tmp,tmpt]=sugfis_mbgd(XTrain,yTrain,{XTune,XTest},{yTune,yTest},'MF',MF,'DR','CDR','nF',nF,'Init','FCM','nMF',nRule,'Opt','AdaBelief','Powerball',Powerball,'DropRule',DropRule,'lr',alpha,'l2',l2,'nIt',nIt,'Nbs',Nbs);
                if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                    [thres{r}(s,id),BestmIter{r}(s,id)]=min(tmpt{1});
                    Bestlr{r}(s,id)=alpha;
                    [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
                end
                times{r}(s,id)=toc;
            end
            for beta=l2s
                tic;
                id=id+1;
                [tmp,tmpt]=sugfis_mbgd(XTrain,yTrain,{XTune,XTest},{yTune,yTest},'MF',MF,'DR','CDR','nF',nF,'Init','FCM','nMF',nRule,'Opt','AdaBelief','Powerball',Powerball,'DropRule',DropRule,'lr',lr,'l2',beta,'nIt',nIt,'Nbs',Nbs);
                if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                    [thres{r}(s,id),BestmIter{r}(s,id)]=min(tmpt{1});
                    Bestl2{r}(s,id)=beta;
                    [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
                end
                times{r}(s,id)=toc;
            end
            for droprule=DropRules
                tic;
                id=id+1;
                [tmp,tmpt]=sugfis_mbgd(XTrain,yTrain,{XTune,XTest},{yTune,yTest},'MF',MF,'DR','CDR','nF',nF,'Init','FCM','nMF',nRule,'Opt','AdaBelief','Powerball',Powerball,'DropRule',droprule,'lr',lr,'l2',l2,'nIt',nIt,'Nbs',Nbs);
                if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                    [thres{r}(s,id),BestmIter{r}(s,id)]=min(tmpt{1});
                    BestDropRule{r}(s,id)=droprule;
                    [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
                end
                times{r}(s,id)=toc;
            end
            for powerball=Powerballs
                tic;
                id=id+1;
                [tmp,tmpt]=sugfis_mbgd(XTrain,yTrain,{XTune,XTest},{yTune,yTest},'MF',MF,'DR','CDR','nF',nF,'Init','FCM','nMF',nRule,'Opt','AdaBelief','Powerball',powerball,'DropRule',DropRule,'lr',lr,'l2',l2,'nIt',nIt,'Nbs',Nbs);
                if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                    [thres{r}(s,id),BestmIter{r}(s,id)]=min(tmpt{1});
                    BestPowerball{r}(s,id)=powerball;
                    [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
                end
                times{r}(s,id)=toc;
            end
            
            
        end
    end
end
save('demoPS.mat','RMSEtrain','RMSEtune','RMSEtest','times','BestmIter','Bestlr','Bestl2','BestDropRule','BestPowerball','lr','l2','DropRule','Powerball','lrs','l2s','DropRules','Powerballs','datasets','nAlgs','Nbs','LN','nRepeats','nRs','thres','LN0','nRs','nIt');

%% Plot results
ids=1:length(LN);
[tmp,ttmp]=deal(nan(length(datasets),length(LN),nRepeats));
for s=1:length(datasets)
    ttmp0=cellfun(@(u)squeeze(u(s,ids)),times,'UniformOutput',false);
    ttmp(s,ids,:)=cat(1,ttmp0{:})';
    for id=1:length(LN)
        tmp(s,id,:)=cell2mat(cellfun(@(u,m)squeeze(u(s,id,m(s,id))),RMSEtest,BestmIter,'UniformOutput',false));
    end
end
lineStyles={'k','k','g','g','b','b','r','r';'-','--','-','--','-','--','-','--'};
close all
Params={log10(lrs),l2s,DropRules,Powerballs};
Rs={'R=2','R=4','R=8','R=16','R=32','R=64'};
for flag=1%:4
    switch flag
        case 1
            idM=1:9;
        case 2
            idM=10:15;
        case 3
            idM=16:25;
        case 4
            idM=26:37;
    end
    f=mat2cell(permute(tmp(:,idM+(1:37:size(tmp,2)-37)',:),[3,2,1]),ones(1,nRepeats));
    f=cellfun(@(x)permute(x,[2,3,1]),f,'UniformOutput',false);
    RR=mat2cell(permute(tmp(:,1,:),[3,2,1]),ones(1,nRepeats));
    RR=cellfun(@(x)permute(x,[2,3,1]),RR,'UniformOutput',false);
    fR=cellfun(@(x,y)reshape(nanmean(x./y,2),[],length(idM)),f,RR,'UniformOutput',false);
    fR=cat(3,fR{:});
    savgRMSE=nanmean(fR,3);
    sstdRMSE=nanstd(fR,[],3);
    figure;
    set(gcf,'DefaulttextFontName','times new roman','DefaultaxesFontName','times new roman','defaultaxesfontsize',10);
    hold on;
    switch flag
        case 1
            for i=1:length(nRs)
                errorbar(1:length(idM), flip(savgRMSE(i,:)),flip(sstdRMSE(i,:)),'Color',lineStyles{1,i},'LineStyle',lineStyles{2,i},'linewidth',2);
            end
            set(gca,'XTick',1:1:length(idM),'XTickLabel',flip(Params{flag}));
            xlabel('$\log_{10}\alpha$','interpreter','latex','fontsize',12);
        case {2,3,4}
            for i=1:length(nRs)
                errorbar(1:length(idM), savgRMSE(i,:),sstdRMSE(i,:),'Color',lineStyles{1,i},'LineStyle',lineStyles{2,i},'linewidth',2);
            end
            set(gca,'XTick',1:1:length(idM),'XTickLabel',Params{flag});
            box on; axis tight;
            switch flag
                case 2
                    xlabel('$l_2$','interpreter','latex','fontsize',12);
                case 3
                    xlabel('$P$','interpreter','latex','fontsize',12);
                case 4
                    xlabel('$\gamma$','interpreter','latex','fontsize',12);
            end
    end
    switch flag
        case 1
            legend(Rs,'FontSize',10,'interpreter','latex','NumColumns',1,'Location','north');
            legend('boxoff')
    end
    ylabel('Average normalized test RMSE');
    set(gca,'yscale','log');
end
