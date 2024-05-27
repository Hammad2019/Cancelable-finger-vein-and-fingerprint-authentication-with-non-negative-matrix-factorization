%%
clc
clear 
close all
%% 
% load Fingerprint Data
dataChest = fullfile('F:\Cooprations\PSU Papers\Implementation Cancelable Matlab Code\Data\Fingerprint Data\Authentication');
imds2 = imageDatastore(dataChest, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Define the number of folds for cross-validation
numFolds = 5;

% Initialize variables to store performance metrics
accuracy = zeros(numFolds, 1);
confusionMatrices = cell(numFolds, 1);

% Create a cross-validation partition for the data
cv = cvpartition(imds2.NumObservations, 'KFold', numFolds);

% Loop through each fold
for fold = 1:numFolds
    % Split the data into training and test sets for the current fold
    trainingIdx = training(cv, fold);
    testIdx = test(cv, fold);
    
    % Select the training and test data for the current fold
    trainingSet1 = subset(imds2, trainingIdx);
    testSet1 = subset(imds2, testIdx);

        %% Create the Fingerprint Deep Model
      layers2 = [imageInputLayer([240 320 1],'Name','FingerPrint')
        convolution2dLayer([3 3],3,'stride',1,'Name','Conv21')
                            maxPooling2dLayer([2 2],'stride',2,'Name','max21')
                            reluLayer('Name','relu21')
                    convolution2dLayer([3 3],3,'stride',1,'numChannels',3,'Name','Conv22')
                    maxPooling2dLayer([2 2],'stride',2,'Name','max22')
                    reluLayer('Name','relu22')
                    convolution2dLayer([3 3],3,'stride',1,'numChannels',3,'Name','Conv23')
                    maxPooling2dLayer([2 2],'stride',2,'Name','max23')
                    reluLayer('Name','relu23')
                    convolution2dLayer([3 3],3,'stride',1,'numChannels',3,'Name','Conv24')
                    maxPooling2dLayer([2 2],'stride',2,'Name','max24')
                    reluLayer('Name','relu24')
                    convolution2dLayer([3 3],3,'stride',1,'numChannels',3,'Name','Conv25')
                    maxPooling2dLayer([2 2],'stride',2,'Name','max25')
               fullyConnectedLayer(1024,'Name','full21') 
          reluLayer('Name','relu25')
          fullyConnectedLayer(200,'Name','full22')
         reluLayer('Name','relu26')
          dropoutLayer(0.7,'Name','drop21')
          fullyConnectedLayer(2,'Name','full23')
          softmaxLayer('Name','soft21')
          classificationLayer('Name','out2')
];

train_options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',3.00000e-04, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',testSet1, ...
    'ValidationFrequency',87, ...
    'Plots','training-progress', ...
    'Verbose',false);


 net2 = trainNetwork(trainingSet1, layers2, train_options);
%%
YPred = classify(net2,testSet1);
end
% Save the Deep Fingerprint Model
save('FingerPrint1.mat','net2');
