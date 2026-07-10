%% 

clear;
clc;
close all;
rng(1);

%% Load saved NASNet result
paths = project_paths();
load(fullfile(paths.models, "nasnet_result.mat"), "trainedNasNet");

%% Load datasets
trainPath = paths.training;
valPath = paths.validation;
testPath = paths.test;


imdsTrain = imageDatastore(trainPath, ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

imdsVal = imageDatastore(valPath, ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

imdsTest = imageDatastore(testPath, ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

%% Resize images for NASNet
inputSize = trainedNasNet.Layers(1).InputSize;

augTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augVal   = augmentedImageDatastore(inputSize(1:2),imdsVal);
augTest  = augmentedImageDatastore(inputSize(1:2),imdsTest);

%% Extract deep features
% global_average_pooling2d_2 gives high-level NASNet features.
featureLayer = "global_average_pooling2d_2";

featuresTrain = activations(trainedNasNet,augTrain,featureLayer,"OutputAs","rows");
featuresVal   = activations(trainedNasNet,augVal,featureLayer,"OutputAs","rows");
featuresTest  = activations(trainedNasNet,augTest,featureLayer,"OutputAs","rows");

YTrain = imdsTrain.Labels;
YVal   = imdsVal.Labels;
YTest  = imdsTest.Labels;

%% Feature selection using MRMR
featureNumbers = [25 50 100 200 300 500 750 1000 1500 2000];
kValues = 1:2:15;

idx = fscmrmr(featuresTrain,YTrain);

%% Tune LDA using validation set
bestValAccLDA = 0;

for n = featureNumbers
    selectedTrain = featuresTrain(:,idx(1:n));
    selectedVal   = featuresVal(:,idx(1:n));

    ldaModel = fitcdiscr(selectedTrain,YTrain);
    YValPred = predict(ldaModel,selectedVal);

    valAcc = mean(YValPred == YVal);

    if valAcc > bestValAccLDA
        bestValAccLDA = valAcc;
        bestNumFeaturesLDA = n;
    end
end

%% Final LDA test
selectedTrain = featuresTrain(:,idx(1:bestNumFeaturesLDA));
selectedTest  = featuresTest(:,idx(1:bestNumFeaturesLDA));

ldaModel = fitcdiscr(selectedTrain,YTrain);
YPred_LDA = predict(ldaModel,selectedTest);

accuracy_LDA = mean(YPred_LDA == YTest);

figure;
confusionchart(YTest,YPred_LDA);
title("NASNet Deep Features + MRMR + LDA");
exportgraphics(gcf, fullfile(paths.figures, "nasnet_mrmr_lda_confusion_matrix.png"));

%% Tune KNN using validation set
bestValAccKNN = 0;

for n = featureNumbers
    selectedTrain = featuresTrain(:,idx(1:n));
    selectedVal   = featuresVal(:,idx(1:n));

    for k = kValues
        knnModel = fitcknn(selectedTrain,YTrain, ...
            "NumNeighbors",k, ...
            "Standardize",true);

        YValPred = predict(knnModel,selectedVal);
        valAcc = mean(YValPred == YVal);

        if valAcc > bestValAccKNN
            bestValAccKNN = valAcc;
            bestNumFeaturesKNN = n;
            bestK = k;
        end
    end
end

%% Final KNN test
selectedTrain = featuresTrain(:,idx(1:bestNumFeaturesKNN));
selectedTest  = featuresTest(:,idx(1:bestNumFeaturesKNN));

knnModel = fitcknn(selectedTrain,YTrain, ...
    "NumNeighbors",bestK, ...
    "Standardize",true);

YPred_KNN = predict(knnModel,selectedTest);

accuracy_KNN = mean(YPred_KNN == YTest);

figure;
confusionchart(YTest,YPred_KNN);
title("NASNet Deep Features + MRMR + KNN");
exportgraphics(gcf, fullfile(paths.figures, "nasnet_mrmr_knn_confusion_matrix.png"));

%% Save results
save(fullfile(paths.models, "nasnet_features_result.mat"), ...
    "featuresTrain","featuresVal","featuresTest", ...
    "idx", ...
    "featureLayer", ...
    "bestNumFeaturesLDA", ...
    "bestNumFeaturesKNN", ...
    "bestValAccLDA", ...
    "bestValAccKNN", ...
    "bestK", ...
    "accuracy_LDA", ...
    "accuracy_KNN", ...
    "ldaModel", ...
    "knnModel", ...
    "YPred_LDA", ...
    "YPred_KNN", ...
    "YTest");

%% Print best parameters
fprintf("\n========== NASNet Best Parameters ==========\n");
fprintf("Feature layer : %s\n", featureLayer);
fprintf("Best LDA Features : %d\n", bestNumFeaturesLDA);
fprintf("Best KNN Features : %d\n", bestNumFeaturesKNN);
fprintf("Best K : %d\n", bestK);
fprintf("Validation Accuracy LDA : %.2f%%\n", bestValAccLDA*100);
fprintf("Validation Accuracy KNN : %.2f%%\n", bestValAccKNN*100);
fprintf("Test Accuracy LDA : %.2f%%\n", accuracy_LDA*100);
fprintf("Test Accuracy KNN : %.2f%%\n", accuracy_KNN*100);
