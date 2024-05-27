clc
clear 
close all
%%
% load Finger vein Data

dataChest = fullfile('F:\Cooprations\PSU Papers\Implementation Cancelable Matlab Code\Data\FV-USM Database\Authentication');
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

    %% Create the Finger vein Deep Model
    layers1 = [imageInputLayer([240 320 1],'Name','FingerVein')
 convolution2dLayer([3 3],3,'stride',1,'Name','Conv_1')
                            maxPooling2dLayer([2 2],'stride',2,'Name','max_1')
                            reluLayer('Name','relu_1')
                    convolution2dLayer([3 3],3,'stride',1,'numChannels',3,'Name','Conv_2')
                    maxPooling2dLayer([2 2],'stride',2,'Name','max_2')
                    reluLayer('Name','relu_2')
                    convolution2dLayer([3 3],3,'stride',1,'numChannels',3,'Name','Conv_3')
                    maxPooling2dLayer([2 2],'stride',2,'Name','max_3')
                    reluLayer('Name','relu_3')
                    convolution2dLayer([3 3],3,'stride',1,'numChannels',3,'Name','Conv_4')
                    maxPooling2dLayer([2 2],'stride',2,'Name','max_4')
               fullyConnectedLayer(1024,'Name','fc_1') 
          reluLayer('Name','relu_4')
          fullyConnectedLayer(200,'Name','fc_2')
          reluLayer('Name','relu_5')
          dropoutLayer(0.7,'Name','drop_1')
          fullyConnectedLayer(2,'Name','fc_3')
          softmaxLayer('Name','soft_1')
          classificationLayer('Name','out1')
];

%%
%%
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

 net1 = trainNetwork(trainingSet1, layers1, train_options);
%%
YPred = classify(net1,testSet1);
end

% Save the Deep Fingerprint Model
save('FingerVein1.mat','net1');