%% Team Name: Body Mass Index (BMI)
% Team member：Shiyue Yang, Junmou Tang, Yange Sun, Crist Lian

function [x, y] = positionEstimator(test_data, modelParams)

    persistent prevSpikeSum prevT prevLastBinCount

    %% ---------------- Constants ----------------
    binSize   = 20;
    startTime = 320;
    maxStep   = 40;
    minStep   = 1.7;
    K_top     = 3;

    startBin = floor(startTime / binSize);

    %% ---------------- Current time/bin ----------------
    currT = size(test_data.spikes, 2);
    currB = floor(currT / binSize);
    decB  = min(max(currB, startBin), modelParams.maxBin);

    isFirstCall = isempty(test_data.decodedHandPos);

    %% ---------------- Feature extraction ----------------
    if isFirstCall
        prevSpikeSum = sum(test_data.spikes, 2);
        prevT = currT;
        cnts = prevSpikeSum;

        b1 = sum(test_data.spikes(:, currT - 2*binSize + 1 : currT - binSize), 2);
        b2 = sum(test_data.spikes(:, currT - binSize + 1   : currT), 2);

        prevLastBinCount = b2;
    else
        b2 = sum(test_data.spikes(:, prevT + 1 : currT), 2);
        cnts = prevSpikeSum + b2;
        b1 = prevLastBinCount;

        prevLastBinCount = b2;
        prevSpikeSum = cnts;
        prevT = currT;
    end

    %% ---------------- LDA classification ----------------
    scores = cnts' * modelParams.ldaW{decB} + modelParams.ldaC{decB};

    if exist('maxk', 'builtin') || exist('maxk', 'file')
        [topScores, topIdx] = maxk(scores, K_top);
    else
        [sortedScores, sortedIdx] = sort(scores, 'descend');
        topScores = sortedScores(1:K_top);
        topIdx = sortedIdx(1:K_top);
    end

    probs = exp(topScores - topScores(1));
    weights = probs / sum(probs);

    pPrior = zeros(2, 1);
    for i = 1:K_top
        pPrior = pPrior + weights(i) * modelParams.meanTraj(topIdx(i), :, decB)';
    end

    if isFirstCall
        x = pPrior(1);
        y = pPrior(2);
        return;
    end

    %% ---------------- Ridge regression prediction ----------------
    feat = [1; sqrt(b1); sqrt(b2)];
    delta = modelParams.regW' * feat;

    prevPos = test_data.decodedHandPos(:, end);
    pLike = prevPos + delta;

    %% ---------------- Trajectory fusion (Sigmoid) ----------------
    denom = modelParams.maxBin - startBin;
    ratio = (decB - startBin) / denom;
    
    k_steep = 8; 
    center = 0.5; 
    alpha = 1 / (1 + exp(k_steep * (ratio - center)));
    
    newPos = alpha * pPrior + (1 - alpha) * pLike; % 融合位置

    %% ---------------- Dynamic angular constraint ----------------
    vCurrent = newPos - prevPos;
    distSq = sum(vCurrent.^2);

    target = modelParams.targetPos(topIdx(1), :)';
    vRef = target - prevPos;
    distRefSq = sum(vRef.^2);

    if distSq > 1e-10 && distRefSq > 1e-10
        dist = sqrt(distSq);
        distRef = sqrt(distRefSq);

        cosTheta = (vCurrent' * vRef) / (dist * distRef);

        if cosTheta < modelParams.cosThresh
            crossSign = sign(vRef(1) * vCurrent(2) - vRef(2) * vCurrent(1));
            if crossSign == 0
                crossSign = 1;
            end

            uRef = vRef / distRef;
            sinVal = crossSign * modelParams.sinThresh;

            uNew = [ ...
                uRef(1) * modelParams.cosThresh - uRef(2) * sinVal;
                uRef(1) * sinVal + uRef(2) * modelParams.cosThresh ...
            ];

            vCurrent = min(dist, minStep) * uNew;
        else
            vCurrent = min(dist, maxStep) * (vCurrent / dist);
        end
    end

    finalPos = prevPos + vCurrent;
    x = finalPos(1);
    y = finalPos(2);
end