function [X,X_test,labels,labels_test,K_train,K_test,NumClasses] = LoadData(DataSet)

tic
disp('Downloading data')
if strcmp(DataSet,'CIFAR100')
    if ~exist('cifar-100-matlab.tar.gz')
        websave('cifar-100-matlab.tar.gz','http://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz');
        gunzip('cifar-100-matlab.tar.gz');untar('cifar-100-matlab.tar');
    end
    load('cifar-100-matlab\train.mat','data','fine_labels')
    X=permute(reshape(data,[size(data,1),32,32,3,]),[3,2,4,1]);
    labels = single(fine_labels)+1; %number labels from 1 to N
    clear data fine_labels
    
    load('cifar-100-matlab\test.mat','data','fine_labels')
    X_test=permute(reshape(data,[size(data,1),32,32,3,]),[3,2,4,1]);
    labels_test = single(fine_labels)+1;

    clear data fine_labels
    NumClasses = 100;
elseif strcmp(DataSet,'CIFAR10')
    if ~exist('cifar-10-matlab.tar.gz')
        websave('cifar-10-matlab.tar.gz','http://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz');
        gunzip('cifar-10-matlab.tar.gz');untar('cifar-10-matlab.tar');
    end
    X = [];
    labels_train = [];
    for i = 1:5
        load(['cifar-10-batches-mat\data_batch_' num2str(i) '.mat'],'data','labels')
        X = [X;data];
        labels_train=[labels_train;labels];
    end
    X=permute(reshape(X,[size(X,1),32,32,3,]),[3,2,4,1]);
    
    load('cifar-10-batches-mat\test_batch.mat','data','labels')
    X_test=permute(reshape(data,[size(data,1),32,32,3,]),[3,2,4,1]);
    labels_test = single(labels)+1;
    clear data labels
    labels = single(labels_train)+1;
    clear data labels_train
    NumClasses = 10;
end
K_train = length(labels);
K_test = length(labels_test);