%% Team Name: Body Mass Index (BMI)
% Team member：Shiyue Yang, Junmou Tang, Yange Sun, Crist Lian

function modelParams = positionEstimatorTraining(training_data)

    % ---------------- Hyperparameters ----------------
    binSize      = 20;
    historyBins  = 2;
    ridgeLambda  = 100;
    ldaLambda    = 1.7;

    angleThreshold = 42;
    cosThresh = cos(angleThreshold * pi / 180);
    sinThresh = sin(angleThreshold * pi / 180);

    % ---------------- Basic dimensions ----------------
    [nTrials,nDirs] = size(training_data);
    nNeurons = 98;

    startBin = 16;

    allT   = arrayfun(@(x) size(x.spikes, 2), training_data);
    allBin = floor(allT / binSize);
    maxBin = max(allBin(:));

    % ---------------- Preallocation ----------------
    trajSum     = zeros(nDirs, 2, maxBin);
    trajCnt     = zeros(nDirs, maxBin);
    finalPosSum = zeros(nDirs, 2);

    totalValidB = sum(max(allBin(:) - startBin, 0));
    featureDim = 1 + nNeurons * historyBins;

    xTrain = zeros(totalValidB, featureDim);
    yTrain = zeros(totalValidB, 2);
    sampleIdx = 1;

    % cumulativeSpikes(dir, trial, neuron, bin)
    cumulativeSpikes = zeros(nDirs, nTrials, nNeurons, maxBin);

    % ---------------- Data preparation ----------------
    for dirIdx = 1:nDirs
        for trialIdx = 1:nTrials
            spikes = training_data(trialIdx, dirIdx).spikes;
            pos    = training_data(trialIdx, dirIdx).handPos(1:2, :);

            finalPosSum(dirIdx, :) = finalPosSum(dirIdx, :) + pos(:, end)';

            nB = floor(size(spikes, 2) / binSize);
            if nB <= startBin
                continue;
            end

            spikesTrimmed = spikes(:, 1:nB * binSize);
            binnedSpikes = squeeze(sum(reshape(spikesTrimmed, nNeurons, binSize, nB), 2));
            if nB == 1
                binnedSpikes = reshape(binnedSpikes, nNeurons, 1);
            end

            cumSpikes = cumsum(binnedSpikes, 2);
            cumulativeSpikes(dirIdx, trialIdx, :, 1:nB) = reshape(cumSpikes, 1, 1, nNeurons, nB);

            if nB < maxBin
                lastCum = reshape(cumSpikes(:, end), 1, 1, nNeurons, 1);
                cumulativeSpikes(dirIdx, trialIdx, :, nB+1:maxBin) = ...
                    repmat(lastCum, 1, 1, 1, maxBin - nB);
            end

            posBin = pos(:, binSize:binSize:nB*binSize);
            trajSum(dirIdx, :, 1:nB) = trajSum(dirIdx, :, 1:nB) + reshape(posBin, 1, 2, nB);
            trajCnt(dirIdx, 1:nB) = trajCnt(dirIdx, 1:nB) + 1;

            for b = (startBin + 1):nB
                featBlock = binnedSpikes(:, b-historyBins+1:b);
                xTrain(sampleIdx, :) = [1, sqrt(featBlock(:))'];
                yTrain(sampleIdx, :) = (posBin(:, b) - posBin(:, b-1))';
                sampleIdx = sampleIdx + 1;
            end
        end
    end

    xTrain = xTrain(1:sampleIdx-1, :);
    yTrain = yTrain(1:sampleIdx-1, :);

    % ---------------- Target position ----------------
    targetPos = finalPosSum ./ nTrials;

    % ---------------- Ridge regression ----------------
    I_ridge = eye(featureDim);
    I_ridge(1,1) = 0;
    regW = (xTrain' * xTrain + ridgeLambda * I_ridge) \ (xTrain' * yTrain);

    % ---------------- Mean trajectory ----------------
    meanTraj = trajSum ./ max(1, reshape(trajCnt, nDirs, 1, maxBin));

    % ---------------- LDA training ----------------
    ldaW = cell(maxBin, 1);
    ldaC = cell(maxBin, 1);

    for b = 1:maxBin
        Xb = squeeze(cumulativeSpikes(:, :, :, b));   % [nDirs x nTrials x nNeurons]
        if nDirs == 1
            Xb = reshape(Xb, 1, nTrials, nNeurons);
        end

        mu = squeeze(mean(Xb, 2));                    % [nDirs x nNeurons]
        if nDirs == 1
            mu = reshape(mu, 1, nNeurons);
        end

        Sigma = zeros(nNeurons, nNeurons);

        for dirIdx = 1:nDirs
            Xk = squeeze(Xb(dirIdx, :, :));          % [nTrials x nNeurons]
            if nTrials == 1
                Xk = reshape(Xk, 1, nNeurons);
            end
            XkCentered = Xk - mu(dirIdx, :);
            Sigma = Sigma + XkCentered' * XkCentered;
        end

        Sigma = Sigma / (nDirs * nTrials - nDirs);
        SigmaReg = Sigma + ldaLambda * eye(nNeurons);

        W = zeros(nNeurons, nDirs);
        C = zeros(1, nDirs);

        for dirIdx = 1:nDirs
            mu_k = mu(dirIdx, :)';
            wk = SigmaReg \ mu_k;
            W(:, dirIdx) = wk;
            C(dirIdx) = -0.5 * (mu_k' * wk);
        end

        ldaW{b} = W;
        ldaC{b} = C;
    end

    % ---------------- Output model ----------------
    modelParams = struct( ...
        'maxBin',    maxBin, ...
        'ldaW',      {ldaW}, ...
        'ldaC',      {ldaC}, ...
        'meanTraj',  meanTraj, ...
        'targetPos', targetPos, ...
        'cosThresh', cosThresh, ...
        'sinThresh', sinThresh, ...
        'regW',      regW ...
    );
end