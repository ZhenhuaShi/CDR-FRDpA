function [RMSEtrain, RMSEtest, AntecedentsB, WB, WAB, WCB, fB] = sugfis_mbgd(XTrain, yTrain, XTest, yTest, varargin)

if nargin < 4
    rng(0)
    XTrain = rand(100, 10);
    yTrain = rand(100, 1);
    XTest = {rand(10, 10), rand(10, 10)};
    yTest = {rand(10, 1), rand(10, 1)};
end

[N, M] = size(XTrain);
if ~iscell(XTest)
    XTest = {XTest};
    yTest = {yTest};
end

if nargin > 4
    [varargin{:}] = convertStringsToChars(varargin{:});
end

% Parse arguments and check if parameter/value pairs are valid
paramNames = {'MF', 'DR', 'nF', 'Init', 'nMF', 'Uncertain', 'TR', ...
    'Opt', 'Powerball', 'DropRule', 'lr', 'l2', 'nIt', 'Nbs', 'RP'};
defaults   = {'Gaussian', 'CDR', 4, 'FCM', 16, 'None', 'Nie-Tan', ...
    'AdaBelief', 0.5, 0.5, 0.01, 0.05, 1000, 64, nan};       % CDR-FCM-RDpA

[MF, DR, nF, Init, nMF, Uncertain, TR, Opt, Powerball, DropRule, lr, l2, nIt, Nbs, RP] ...
    = internal.stats.parseArgs(paramNames, defaults, varargin{:});

if Nbs > N
    Nbs = N;
end
if nF > M
    nF = M;
end
if strcmp(Init, 'GP')
    if nMF == 0
        nMF = 2;
    end
    nRule = nMF^nF;
else
    nRule = nMF;
    nMF = 0;
end
AdaptiveDropRule=0;
if RP(1)==0
    AdaptiveDropRule=1;
    RP=RP(2:end);
end
nRuleStep = nRule / 2 / length(RP);

if ismember(Init, {'GP', 'Grid Partitioning'}) && ismember(DR, {'TLP1', 'TLP2', 'TLPx'})
    error('Init-DR conflict.')
end
if ismember(Init, {'GP', 'Grid Partitioning'}) && ismember(DR, {'None'}) && M > nF
    error('Init-DR conflict.')
end

switch DR
    case {'None'}
        XTrainA = XTrain;
        [WA0, WC0] = deal([zeros(1, M); eye(M)]);
        [MA, MC] = deal(M);
        nF = M;
    case {'PCA', 'CDR'} % InitPCA==CDR
        [WPCA, XTrainA] = pca(XTrain, 'NumComponents', nF);
        [WA0, WC0] = deal([zeros(1, nF); WPCA]);
        [MA, MC] = deal(nF);
    case {'TLP1'}
        [WPCA, XTrainA] = pca(XTrain, 'NumComponents', nF);
        XTrainA = [XTrainA, XTrain];
        WA0 = [zeros(1, nF); WPCA];
        WC0 = [zeros(1, M); eye(M)];
        [MA, MC] = deal(nF+M, M);
    case {'TLP2', 'TLPx'}
        [WPCA, XTrainA] = pca(XTrain, 'NumComponents', nF);
        XTrainA = [XTrainA, XTrain];
        [WA0, WC0] = deal([zeros(1, nF); WPCA]);
        [MA, MC] = deal(nF+M);
    case {'DR'}
        [WPCA, XTrainA] = pca(XTrain, 'NumComponents', nF);
        WA0 = [zeros(1, nF); WPCA];
        WC0 = [zeros(1, M); eye(M)];
        [MA, MC] = deal(nF, M);
    case {'Rand','initRand'}
        WPCA= orth(rand(M,nF));
        XTrainA = XTrain * WPCA;
        [WA0, WC0] = deal([zeros(1, nF); WPCA]);
        [MA, MC] = deal(nF);
    case {'PCAxy','initPCAxy'}
        WPCA = pca([XTrain yTrain], 'NumComponents', nF);
        WPCA = WPCA(1:end-1,:);
        XTrainA = XTrain * WPCA;
        [WA0, WC0] = deal([zeros(1, nF); WPCA]);
        [MA, MC] = deal(nF);
end

switch Init
    case {'Rand'}
        C0 = 2 * rand(nRule, MA) - 1;
        Sigma0 = 5 * rand(nRule, MA);
        W0 = 2 * rand(nRule, MC+1) - 1;
    case {'Grid Partitioning', 'GP'}
        C0 = zeros(nF, nMF);
        Sigma0 = C0;
        W0 = zeros(nRule, MC+1);
        for m = 1:nF % Initialization
            C0(m, :) = linspace(min(XTrainA(:, m)), max(XTrainA(:, m)), nMF);
            Sigma0(m, :) = std(XTrainA(:, m));
        end
    case {'FCMx', 'FCM', 'LogFCM', 'HFCM', 'LogHFCM'}
        W0 = zeros(nRule, MC+1); % Rule consequents
        % FCM initialization
        [C0, U] = FuzzyCMeans(XTrainA, nRule, [2, 100, 0.001, 0]);
        Sigma0 = C0;
        for ir = 1:nRule
            Sigma0(ir, :) = std(XTrainA, U(ir, :));
            W0(ir, 1) = U(ir, :) * yTrain / sum(U(ir, :));
        end
        if ismember(Init, {'HFCM', 'LogHFCM'})
            Sigma0 = sqrt(size(XTrainA, 2)) * Sigma0;
        end
    case {'FCMy'}
        W0 = zeros(nRule, MC+1); % Rule consequents
        % FCM initialization
        [W0(:, 1), U] = FuzzyCMeans(yTrain, nRule, [2, 100, 0.001, 0]);
        C0 = U * XTrainA;
        Sigma0 = C0;
        for ir = 1:nRule
            Sigma0(ir, :) = std(XTrainA, U(ir, :));
        end
    case {'kMx'}
        W0 = zeros(nRule, MC+1); % Rule consequents
        [ids, C0] = kmeans(XTrainA, nRule, 'replicate', 3);
        Sigma0 = C0;
        for ir = 1:nRule
            Sigma0(ir, :) = std(XTrainA(ids == ir, :));
            W0(ir, 1) = mean(yTrain(ids == ir));
        end
    case {'kMy'}
        W0 = zeros(nRule, MC+1); % Rule consequents
        [ids, W0(:, 1)] = kmeans(yTrain, nRule, 'replicate', 3);
        [C0, Sigma0] = deal(zeros(nRule, MA));
        for ir = 1:nRule
            C0(ir, :) = mean(XTrainA(ids == ir, :));
            Sigma0(ir, :) = std(XTrainA(ids == ir, :));
        end
end
Sigma0(Sigma0 == 0) = mean(Sigma0(:));
switch Uncertain
    case {'None'}
        switch MF
            case {'Bell-Shaped', 'gbell'}
                Antecedents0 = cat(3, Sigma0, 5*ones(size(C0)), C0);
            case {'Gaussian', 'gauss'}
                Antecedents0 = cat(3, Sigma0, C0);
            case {'Trapezoidal', 'trap'}
                [A0, D0] = deal(C0-10*Sigma0, C0+10*Sigma0);
                [B0, C0] = deal(C0-.5*Sigma0, C0+.5*Sigma0);
                Antecedents0 = cat(3, A0, B0, C0, D0);
            case {'Triangular', 'tri'}
                [A0, D0] = deal(C0-10*Sigma0, C0+10*Sigma0);
                Antecedents0 = cat(3, A0, C0, D0);
        end
    case {'Mean', 'mean'}
        switch MF
            case {'Gaussian', 'gauss'}
                Antecedents0 = cat(3, Sigma0, C0-0.01, C0+0.01);
        end
    case {'Variance', 'var'}
        switch MF
            case {'Gaussian', 'gauss'}
                Antecedents0 = cat(3, Sigma0-0.01, Sigma0+0.01, C0);
        end
end
[Antecedents, W, WA, WC] = deal(Antecedents0, W0, WA0, WC0);
[AntecedentsB, WB, WAB, WCB] = deal(Antecedents0, W0, WA0, WC0);
fB = zeros(1, nRule);

%% Iterative update
[beta1, beta2, thre, sp] = deal(0.9, 0.999, inf, 10^(-8));
RMSEtrain = zeros(1, nIt);
RMSEtest = cellfun(@(u)RMSEtrain, XTest, 'UniformOutput', false);
yPred = nan(Nbs, 1);
iit = 0;
for it = 1:nIt
    iit = iit + 1;
    if it == 1 || ismember(it, RP)
        switch DR
            case {'None', 'PCA', 'Rand', 'PCAxy'}
                [mea, var] = deal({0, 0});
            case {'CDR', 'initRand', 'initPCAxy', 'DR', 'TLP1', 'TLP2'}
                [mea, var] = deal({0, 0, 0});
            case {'TLPx'}
                [mea, var] = deal({0, 0, 0, 0});
        end
        if ismember(it, RP)
            if AdaptiveDropRule
                DropRule=nRule*DropRule/(nRule - nRuleStep);
            end
            nRule = nRule - nRuleStep;
            [~, idx] = maxk(fB, nRule);
            Antecedents = Antecedents(idx, :, :, :);
            W = W(idx, :);
            iit = 1;
            fB = zeros(1, nRule);
            thre = inf;
        end
    end
    deltaA = zeros(size(Antecedents));
    deltaW = l2 * [zeros(size(W, 1), 1), W(:, 2:end)]; % consequent
    [deltaXA, deltaXC] = deal(zeros(Nbs, nF));
    idsTrain = datasample(1:N, Nbs, 'replace', false);
    [XTraina, XTrainc] = deal(XTrain(idsTrain, :));

    switch DR
        case {'None', 'PCA', 'Rand', 'PCAxy', 'CDR', 'initRand', 'initPCAxy', 'DR'}
            XTrainA = [ones(Nbs, 1), XTraina] * WA;
            XTrainC = [ones(Nbs, 1), XTrainc] * WC;
        case {'TLP1'}
            XTrainA = [[ones(Nbs, 1), XTraina] * WA, XTrain(idsTrain, :)];
            XTrainC = [ones(Nbs, 1), XTrainc] * WC;
        case {'TLP2'}
            XTrainA = [[ones(Nbs, 1), XTraina] * WA, XTrain(idsTrain, :)];
            XTrainC = [[ones(Nbs, 1), XTrainc] * WC, XTrain(idsTrain, :)];
        case {'TLPx'}
            XTrainA = [[ones(Nbs, 1), XTraina] * WA, XTrain(idsTrain, :)];
            XTrainC = [[ones(Nbs, 1), XTrainc] * WC, XTrain(idsTrain, :)];
    end

    for n = 1:Nbs
        idsKeep = rand(1, nRule) <= DropRule;
        [fKeep, deltamu] = CalculateMF(XTrainA(n, :), Antecedents, idsKeep, MF, nMF, Uncertain);
        if sum(~isfinite(fKeep(:)))
            continue;
        end
        if ~sum(fKeep(:)) % special case: all f(n,:)=0; no dropRule
            idsKeep = true(1, nRule);
            [fKeep, deltamu] = CalculateMF(XTrainA(n, :), Antecedents, idsKeep, MF, nMF, Uncertain);
        end
        yR = [1, XTrainC(n, :)] * W(idsKeep, :)';
        switch Uncertain
            case {'None'}
                if ismember(Init, {'LogFCM', 'LogHFCM'})
                    fKeep = -1 ./ log(fKeep);
                    fBar = fKeep / sum(fKeep);
                    yPred(n) = yR * fBar; % prediction
                    deltaYmu = (yPred(n) - yTrain(idsTrain(n))) * (yR * sum(fKeep) - yR * fKeep) / sum(fKeep)^2 .* (fKeep.^2)';
                else
                    fBar = fKeep / sum(fKeep);
                    yPred(n) = yR * fBar; % prediction
                    % Compute delta
                    deltaYmu = (yPred(n) - yTrain(idsTrain(n))) * (yR * sum(fKeep) - yR * fKeep) / sum(fKeep)^2 .* fKeep';
                end
            case {'Mean', 'mean', 'Variance', 'var'}
                switch TR
                    case {'Karnik-Mendel', 'km'}
                        [syR, iyR] = sort(yR);
                        sfl = fKeep(iyR, 1);
                        sfr = fKeep(iyR, 2);
                        sfly = fKeep(iyR, 1) .* syR';
                        sfry = fKeep(iyR, 2) .* syR';
                        [ylPred, il] = min((cumsum(sfry)+flip(cumsum(flip(sfly))))./(cumsum(sfr) + flip(cumsum(flip(sfl)))));
                        [yrPred, ir] = max((cumsum(sfly)+flip(cumsum(flip(sfry))))./(cumsum(sfl) + flip(cumsum(flip(sfr)))));
                        yPred(n) = (ylPred + yrPred) / 2;
                        fBar = zeros(size(fKeep, 1), 1);
                        fBar(iyR) = ([sfr(1:il); sfl(il + 1:end)] / (sum(sfr(1:il)) + sum(sfl(il + 1:end))) + [sfl(1:ir); sfr(ir + 1:end)] / (sum(sfl(1:ir)) + sum(sfr(ir + 1:end)))) / 2;
                        deltaf = zeros(size(fKeep));
                        deltaf(iyR(1:il), 2) = (syR(1:il) - ylPred) / (sum(sfr(1:il)) + sum(sfl(il + 1:end)));
                        deltaf(iyR(il + 1:end), 1) = (syR(il + 1:end) - ylPred) / (sum(sfr(1:il)) + sum(sfl(il + 1:end)));
                        deltaf(iyR(1:ir), 1) = deltaf(iyR(1:ir), 1) + (syR(1:ir)' - yrPred) / (sum(sfl(1:ir)) + sum(sfr(ir + 1:end)));
                        deltaf(iyR(ir + 1:end), 2) = deltaf(iyR(ir + 1:end), 1) + (syR(ir + 1:end)' - yrPred) / (sum(sfl(1:ir)) + sum(sfr(ir + 1:end)));
                        deltaYmu = (yPred(n) - yTrain(idsTrain(n))) / 2 * deltaf .* fKeep;
                    case {'Nie-Tan'}
                        fAvg = (fKeep(:, 1) + fKeep(:, 2)) / 2;
                        fBar = fAvg / sum(fAvg);
                        yPred(n) = yR * fBar;
                        deltaYmu = (yPred(n) - yTrain(idsTrain(n))) / 2 * (yR' * sum(fAvg) - yR * fAvg) ./ sum(fAvg).^2 .* fKeep;
                end
        end
        if ~sum(~isfinite(deltaYmu(:)))
            deltaW(idsKeep, :) = deltaW(idsKeep, :) + (yPred(n) - yTrain(idsTrain(n))) * fBar * [1, XTrainC(n, :)];
            if ismember(DR, {'CDR', 'initRand', 'initPCAxy', 'TLP2', 'TLPx'})
                deltaXC(n, :) = (yPred(n) - yTrain(idsTrain(n))) * fBar' * W(idsKeep, 2:1+nF);
            end
            switch Uncertain
                case {'None'}
                    if ~strcmp(Init, 'GP')
                        deltaA(idsKeep, :, :) = deltaA(idsKeep, :, :) + deltaYmu' .* deltamu;
                        if ismember(DR, {'CDR', 'initRand', 'initPCAxy', 'DR', 'TLP1', 'TLP2', 'TLPx'})
                            switch MF
                                case {'Bell-Shaped', 'gbell', 'Gaussian', 'gauss'}
                                    deltaXA(n, :) = -sum(deltaYmu'.*deltamu(:, 1:nF, end), 1);
                                case {'Trapezoidal', 'trap'}
                                    deltaXA(n, :) = -sum(deltaYmu'.*(deltamu(:, 1:nF, 2) + deltamu(:, 1:nF, 3)), 1);
                                case {'Triangular', 'tri'}
                                    deltaXA(n, :) = -sum(deltaYmu'.*deltamu(:, 1:nF, 2), 1);
                            end
                        end
                    else
                        deltaA = deltaA + permute(sum(deltaYmu .* deltamu, 2), [1, 3, 4, 2]);
                        if ismember(DR, {'CDR', 'initRand', 'initPCAxy', 'DR'})
                            switch MF
                                case {'Bell-Shaped', 'gbell', 'Gaussian', 'gauss'}
                                    deltaXA(n, :) = -sum(permute(sum(deltaYmu .* deltamu(:, :, :, end), 2), [1, 3, 4, 2]), 2);
                                case {'Trapezoidal', 'trap'}
                                    deltaXA(n, :) = -sum(permute(sum(deltaYmu .* (deltamu(:, :, :, 2) + deltamu(:, :, :, 3)), 2), [1, 3, 4, 2]), 2);
                                case {'Triangular', 'tri'}
                                    deltaXA(n, :) = -sum(permute(sum(deltaYmu .* deltamu(:, :, :, 2), 2), [1, 3, 4, 2]), 2);
                            end
                        end
                    end
                case {'Mean', 'mean', 'Variance', 'var'}
                    if ~strcmp(Init, 'GP')
                        deltaA(idsKeep, :, :) = deltaA(idsKeep, :, :) + permute(sum(deltaYmu .* deltamu, 2), [1, 3, 4, 2]);
                    else
                        deltaA = deltaA + permute(sum(sum(permute(deltaYmu, [3, 1, 2]) .* deltamu, 2), 3), [1, 4, 5, 2, 3]);
                    end
            end
        end
    end
    switch DR
        case {'None', 'PCA', 'Rand', 'PCAxy', 'PCA2', 'LDA', 'SDSPCA', 'SDSPCAAN'}
            params = {Antecedents, W};
            delta = {deltaA, deltaW};
        case {'DR', 'TLP1'}
            deltaWA0 = [ones(Nbs, 1), XTraina]' * deltaXA;
            if ismember(DR, {'TLP1'})
                deltaWA = deltaWA0;
            elseif ismember(DR, {'DR'})
                deltaWA = l2 * [zeros(1, size(WA, 2)); WA(2:end, :)] + deltaWA0;
            end
            params = {Antecedents, W, WA};
            delta = {deltaA, deltaW, deltaWA};
        case {'CDR', 'initRand', 'initPCAxy', 'TLP2'}
            deltaWA = [ones(Nbs, 1), XTraina]' * deltaXA;
            deltaWC = [ones(Nbs, 1), XTrainc]' * deltaXC;
            deltaWA = l2 * [zeros(1, size(WA, 2)); WA(2:end, :)] + deltaWA + deltaWC;
            params = {Antecedents, W, WA};
            delta = {deltaA, deltaW, deltaWA};
        case {'TLPx'}
            deltaWA = [ones(Nbs, 1), XTraina]' * deltaXA;
            deltaWC = [ones(Nbs, 1), XTrainc]' * deltaXC;
            params = {Antecedents, W, WA, WC};
            delta = {deltaA, deltaW, deltaWA, deltaWC};
    end

    % powerball
    delta = cellfun(@(deltaA)sign(deltaA).*(abs(deltaA).^Powerball), delta, 'UniformOutput', false);
    switch Opt
        case {'AdaBound'}
            lb = lr * (1 - 1 / ((1 - beta2) * iit + 1));
            ub = lr * (1 + 1 / ((1 - beta2) * iit));
            mea = cellfun(@(deltaA, mA)beta1*mA+(1 - beta1)*deltaA, delta, mea, 'UniformOutput', false);
            var = cellfun(@(deltaA, vA)beta2*vA+(1 - beta2)*deltaA.^2, delta, var, 'UniformOutput', false);
            mHat = cellfun(@(mA)mA/(1 - beta1^iit), mea, 'UniformOutput', false);
            vHat = cellfun(@(vA)vA/(1 - beta2^iit), var, 'UniformOutput', false);
            lrb = cellfun(@(vAHat)min(ub, max(lb, lr ./ (sqrt(vAHat) + sp))), vHat, 'UniformOutput', false);
            params = cellfun(@(A, lrA, mAHat)A-lrA.*mAHat, params, lrb, mHat, 'UniformOutput', false);
        case {'SGDM'}
            mea = cellfun(@(deltaA, mA)beta1*mA+deltaA, delta, mea, 'UniformOutput', false);
            params = cellfun(@(A, mA)A-lr*mA, params, mea, 'UniformOutput', false);
        case {'Adam'}
            mea = cellfun(@(deltaA, mA)beta1*mA+(1 - beta1)*deltaA, delta, mea, 'UniformOutput', false);
            var = cellfun(@(deltaA, vA)beta2*vA+(1 - beta2)*deltaA.^2, delta, var, 'UniformOutput', false);
            mHat = cellfun(@(mA)mA/(1 - beta1^iit), mea, 'UniformOutput', false);
            vHat = cellfun(@(vA)vA/(1 - beta2^iit), var, 'UniformOutput', false);
            params = cellfun(@(A, mAHat, vAHat)A-lr*mAHat./(sqrt(vAHat) + sp), params, mHat, vHat, 'UniformOutput', false);
        case {'AdaBelief'}
            mea = cellfun(@(deltaA, mA)beta1*mA+(1 - beta1)*deltaA, delta, mea, 'UniformOutput', false);
            var = cellfun(@(deltaA, vA, mA)beta2*vA+(1 - beta2)*(deltaA - mA).^2, delta, var, mea, 'UniformOutput', false);
            mHat = cellfun(@(mA)mA/(1 - beta1^iit), mea, 'UniformOutput', false);
            vHat = cellfun(@(vA)vA/(1 - beta2^iit), var, 'UniformOutput', false);
            params = cellfun(@(A, mAHat, vAHat)A-lr*mAHat./(sqrt(vAHat) + sp), params, mHat, vHat, 'UniformOutput', false);
    end
    switch DR
        case {'None', 'PCA', 'Rand', 'PCAxy'}
            [Antecedents, W] = deal(params{1}, params{2});
        case {'CDR', 'initRand', 'initPCAxy', 'DR', 'TLP1', 'TLP2'}
            [Antecedents, W, WA] = deal(params{1}, params{2}, params{3});
            if ismember(DR, {'CDR', 'TLP2'})
                WC = WA;
            end
        case {'TLPx'}
            [Antecedents, W, WA, WC] = deal(params{1}, params{2}, params{3}, params{4});
    end

    % Training RMSE on the minibatch
    yPred(isnan(yPred)) = nanmean(yPred);
    RMSEtrain(it) = sqrt(sum((yTrain(idsTrain)-yPred).^2)/Nbs);
    % Test RMSE
    for i = 1:length(XTest)
        NTest = size(XTest{i}, 1);
        [XTesta, XTestc] = deal(XTest{i});
        switch DR
            case {'None', 'PCA', 'Rand', 'PCAxy', 'CDR', 'initRand', 'initPCAxy', 'DR'}
                XTestA = [ones(NTest, 1), XTesta] * WA;
                XTestC = [ones(NTest, 1), XTestc] * WC;
            case {'TLP1'}
                XTestA = [[ones(NTest, 1), XTesta] * WA, XTesta];
                XTestC = [ones(NTest, 1), XTestc] * WC;
            case {'TLP2'}
                XTestA = [[ones(NTest, 1), XTesta] * WA, XTesta];
                XTestC = [[ones(NTest, 1), XTestc] * WC, XTestc];
            case {'TLPx'}
                XTestA = [[ones(NTest, 1), XTesta] * WA, XTesta];
                XTestC = [[ones(NTest, 1), XTestc] * WC, XTestc];
        end
        yR = [ones(NTest, 1), XTestC] * W';
        idsKeep = true(1, nRule);
        idsKeep(DropRule == 0) = 0;
        switch Uncertain
            case {'None'}
                fKeep = zeros(NTest, nRule); % firing level of rules
                for n = 1:NTest
                    fKeep(n, :) = CalculateMF(XTestA(n, :), Antecedents, idsKeep, MF, nMF, Uncertain);
                end
                if ismember(Init, {'LogFCM', 'LogHFCM'})
                    fKeep = -1 ./ log(fKeep);
                end
                yPredTest = sum(fKeep.*yR, 2) ./ sum(fKeep, 2); % prediction
            case {'Mean', 'mean', 'Variance', 'var'}
                switch TR
                    case {'Karnik-Mendel', 'km'}
                        fKeep = zeros(NTest, nRule, 2); % firing level of rules
                        yPredTest = nan(NTest, 1);
                        for n = 1:NTest
                            fKeep(n, :, :) = CalculateMF(XTestA(n, :), Antecedents, idsKeep, MF, nMF, Uncertain);
                            [syR, iyR] = sort(yR(n, :));
                            sfl = fKeep(n, iyR, 1);
                            sfr = fKeep(n, iyR, 2);
                            sfly = fKeep(n, iyR, 1) .* syR;
                            sfry = fKeep(n, iyR, 2) .* syR;
                            ylPred = min((cumsum(sfry)+flip(cumsum(flip(sfly))))./(cumsum(sfr) + flip(cumsum(flip(sfl)))));
                            yrPred = max((cumsum(sfly)+flip(cumsum(flip(sfry))))./(cumsum(sfl) + flip(cumsum(flip(sfr)))));
                            yPredTest(n) = (ylPred + yrPred) / 2;
                        end
                    case {'Nie-Tan'}
                        fKeep = zeros(NTest, nRule, 2); % firing level of rules
                        yPredTest = nan(NTest, 1);
                        for n = 1:NTest
                            fKeep(n, :, :) = CalculateMF(XTestA(n, :), Antecedents, idsKeep, MF, nMF, Uncertain);
                            fAvg = (fKeep(n, :, 1) + fKeep(n, :, 2)) / 2;
                            fBar = fAvg / sum(fAvg);
                            yPredTest(n) = yR(n, :, :) * fBar';
                        end
                end
        end
        yPredTest(isnan(yPredTest)) = nanmean(yPredTest);
        RMSEtest{i}(it) = sqrt((yTest{i}-yPredTest)'*(yTest{i} - yPredTest)/NTest);
        if isnan(RMSEtest{i}(it)) && it > 1
            RMSEtest{i}(it) = RMSEtest{i}(it - 1);
        end
        if nargout > 2 && i == 1 && RMSEtest{i}(it) < thre
            thre = RMSEtest{i}(it);
            [AntecedentsB, WB, WAB, WCB, fB] = deal(Antecedents, W, WA, WC, mean(mean(fKeep, 3)));
        end
    end
end
if length(XTest) == 1
    RMSEtest = RMSEtest{1};
end
end