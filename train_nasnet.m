%%
clear;
clc;
close all;
rng(1)
%% Load datasets

paths = project_paths();
trainPath = paths.training;
valPath = paths.validation;
testPath = paths.test;


imdsTrain = imageDatastore(trainPath,...
    "IncludeSubfolders",true,...
    "LabelSource","foldernames");

imdsVal = imageDatastore(valPath,...
    "IncludeSubfolders",true,...
    "LabelSource","foldernames");

imdsTest = imageDatastore(testPath,...
    "IncludeSubfolders",true,...
    "LabelSource","foldernames");


disp("Training set:");
countEachLabel(imdsTrain)

disp("Validation set:");
countEachLabel(imdsVal)

disp("Test set:");
countEachLabel(imdsTest)


%% Load the pretrained NasNet network

net = nasnetlarge;


inputSize = net.Layers(1).InputSize;

numClasses = numel(categories(imdsTrain.Labels));
% analyzeNetwork(net)

%% Resize images for NasNet

augmenter = imageDataAugmenter( ...
    "RandRotation",[-10 10], ...
    "RandXTranslation",[-10 10], ...
    "RandYTranslation",[-10 10], ...
    "RandXReflection",true);

augTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    "DataAugmentation",augmenter);

augVal = augmentedImageDatastore(inputSize(1:2),imdsVal);

augTest = augmentedImageDatastore(inputSize(1:2),imdsTest);


%% Modify the pretrained network

lgraph = layerGraph(net);

newFC = fullyConnectedLayer(numClasses,...
    "Name","predictions",...
    "WeightLearnRateFactor",10,...
    "BiasLearnRateFactor",10);

newOutput = classificationLayer("Name","ClassificationLayer_predictions");

lgraph = replaceLayer(lgraph,"predictions",newFC);
lgraph = replaceLayer(lgraph,"ClassificationLayer_predictions",newOutput);

%analyzeNetwork(lgraph)


%% Define the training options

if canUseGPU
    execEnv = "gpu";
else
    execEnv = "cpu";
end

disp("Execution environment:");
disp(execEnv);

options = trainingOptions("adam", ...
    "MaxEpochs",5, ...
    "MiniBatchSize",4, ...
    "InitialLearnRate",1e-4, ...
    "ValidationData",augVal, ...
    "ValidationFrequency",40, ...
    "ValidationPatience", 8, ...
    "Shuffle","every-epoch", ...
    "ExecutionEnvironment",execEnv, ...
    "Plots","training-progress", ...
    "Verbose",true);

%% Train the modified NasNet network

trainedNasNet = trainNetwork(augTrain,lgraph,options);


%%Test the trained model


YPredNasNet = classify(trainedNasNet,augTest);

YTest = imdsTest.Labels;

accuracyNasNet = mean(YPredNasNet == YTest);
fprintf("NASNet test accuracy: %.2f%%\n", accuracyNasNet * 100);


%% Display the confusion matrix

figure
confusionchart(YTest,YPredNasNet)

title("NasNet Confusion Matrix")

exportgraphics(gcf, fullfile(paths.figures, "nasnet_confusion_matrix.png"));

%%save

save(fullfile(paths.models, "nasnet_result.mat"), ...
    "trainedNasNet", "accuracyNasNet", "YPredNasNet", "YTest");
