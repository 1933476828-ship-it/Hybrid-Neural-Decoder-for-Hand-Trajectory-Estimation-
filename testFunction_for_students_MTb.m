% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 



function RMSE = testFunction_for_students_MTb(Demo_2)

load monkeydata_training.mat

% Set random number generator
rng(2013);
ix = randperm(length(trial));

% addpath("BMI_KNN");
addpath("BMI");

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);
% trainingData = trial(ix(51:end),:);
% testData = trial(ix(1:50),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  

figure
hold on
axis square
grid

% 训练模型并记录时间
disp('Training model...');
tic; % 开启计时器 (Start stopwatch timer)
modelParameters = positionEstimatorTraining(trainingData);


for tr=1:size(testData,1)
    % display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            
        end
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:), decodedHandPos(2,:), 'Color', '#5471ae', 'LineWidth', 1.5);
        plot(testData(tr,direc).handPos(1,times), testData(tr,direc).handPos(2,times), 'Color', '#f1b9be', 'LineWidth', 1.5);
    end
end

legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions) 

% rmpath(genpath("BMI_KNN"))
% rmpath(genpath("Body Mass Index (BMI)"))

trainingTime = toc; % 停止计时并获取经过的时间 (Read stopwatch timer)
fprintf('Whole process completed in %.4f seconds.\n\n', trainingTime);


end
