% Clear all open windows and all parameters
clc
clear 
close all
%%
% Load and preprocess your Test data
% Make sure it has the same size as the training data
 data1 = fullfile('F:\Cooprations\PSU Papers\Implementation Cancelable Matlab Code\Data\Fingerprint Data\Authentication');
 data2 = fullfile('F:\Cooprations\PSU Papers\Implementation Cancelable Matlab Code\Data\FV-USM Database\Authentication');

 imds1 = imageDatastore(data1, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
imds2 = imageDatastore(data2, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
    [testSet,trainingSet] = splitEachLabel(imds1, 0.2, 'randomize');
    [testSet1,trainingSet1] = splitEachLabel(imds2, 0.2, 'randomize');


Labels_fingervein=trainingSet.Labels;
Labels_fingerprint=trainingSet1.Labels;

combinedLabels= [Labels_fingervein;Labels_fingerprint];
numClass=numel(countcats(combinedLabels));

% Resize and preprocess test images (similar to training)
inputSize = [240 320 1];
imds_augmented1 = augmentedImageDatastore(inputSize, testSet, 'ColorPreprocessing', 'gray2rgb');
imds_augmented2 = augmentedImageDatastore(inputSize, testSet1, 'ColorPreprocessing', 'gray2rgb');

% Load Features 
load('FingerVein1.mat');
load('FingerPrint1.mat');

% Extract features from the data
FV_features  = net1.Layers(13, 1).Bias; %Extract the Deep Features of Finger Vein from the output of the First Fully connected layer
FP_features  = net1.Layers(16, 1).Bias; %Extract the Deep Features of Fingerprint from the output of the First Fully connected layer
Combined_Features = cat(1, FV_features,FP_features);

% Apply NMF to test data using the same 'W' obtained from training
[W, H] = nnmf_custom(Combined_Features, 1);
W= nonzeros(W);

% Generate Cancelable Template
cancelable_templates = generate_cancelable_templates(W);
Cancelable_Value = convert_hash_to_numeric_matrix(cancelable_templates);


lgraph = layerGraph;
lgraph = addLayers(lgraph,net1.Layers(1:14));
lgraph = addLayers(lgraph,net2.Layers(1:17));

      layer3 =[
           concatenationLayer(2,2,"Name","Concat")
      fullyConnectedLayer(Cancelable_Value,'Name','Cancelable');
            reluLayer('Name','Relu_Combined')
                dropoutLayer(0.2,'Name','drop_Combined')
          fullyConnectedLayer(2,'Name','Full_Combined')
          softmaxLayer('Name','soft_Combined')
          classificationLayer('Name','classification')
];
          
 lgraph = addLayers(lgraph,layer3);
 

 lgraph = connectLayers(lgraph,"fc_2","concat/in1");
lgraph = connectLayers(lgraph,"full22","concat/in2");
 
figure
plot(lgraph)
%%
%%
train_options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',3.0000000e-04, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imds_augmented1, ...
    'ValidationFrequency',87, ...
    'Plots','training-progress', ...
    'Verbose',false);

% Training the test data using the Network

net = trainNetwork(imds_augmented1, lgraph, train_options);

% Classify the test data using the trained classifier model

YPred = classify(net,imds_augmented1);


analyzeNetwork(net)









