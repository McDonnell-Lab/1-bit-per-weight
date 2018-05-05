clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This script written by Mark McDonnell, February 2018
%
%Paper: Training wide residual networks for deployment using a single bit for each weight
%Author: M. D. McDonnell
%Contact: mark.mcdonnell@unisa.edu.au
%This code was used for experiments described in a paper  published in ICLR 2018 (iclr.cc).
%The ICLR version of the paper and the double-blind open peer review can be found at
%https://openreview.net/forum?id=rytNfI1AZ (download the PDF here: https://openreview.net/pdf?id=rytNfI1AZ )
%The paper is also on arxiv: https://arxiv.org/abs/1802.08530

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This script demonstrates how to run our trained 1-bit-per-weight model on the CIFAR-10 test set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%In this script, no GPU is assumed, so the CPU runs the model and the total run time comes to about 15 minutes
%On GPU, inference is of course much faster

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 0: select which data set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DataSet = 1; %CIFAR10
DataSet = 2; %CIFAR100

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 1 %download and load CIFAR-10 or CIFAR test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if DataSet == 1
    if ~exist('cifar-10-matlab.tar.gz')
        websave('cifar-10-matlab.tar.gz','http://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz');
        gunzip('cifar-10-matlab.tar.gz');untar('cifar-10-matlab.tar');
    end
    load('cifar-10-batches-mat\test_batch.mat','data','labels')
    TestImages=permute(reshape(data,[size(data,1),32,32,3,]),[3,2,4,1]);
    labels_test = single(labels)+1;
    clear data labels
    NumClasses = 10;
elseif DataSet == 2
    if ~exist('cifar-100-matlab.tar.gz')
        websave('cifar-100-matlab.tar.gz','http://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz');
        gunzip('cifar-100-matlab.tar.gz');untar('cifar-100-matlab.tar');
    end
    load('cifar-100-matlab\test.mat','data','fine_labels')
    TestImages=permute(reshape(data,[size(data,1),32,32,3,]),[3,2,4,1]);
    labels_test = single(fine_labels)+1;
    clear data fine_labels
    NumClasses = 100;
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step 2: load the trained model and convert the weights stored using a single bit to single precision
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this is the wide Resnet 20-10 with cutout augmentation model. Error rate = 3.82%
if DataSet == 1
    disp('Running trained 1-bit-per-weight 20-10 Wide ResNet on the CIFAR10 test set')
    load('TrainedWideResNet20-10_CIFAR10_1-bit-per-weight','BinaryWeights','Moments','BNG','BNB','LayerWeights','Stride','Padding')
elseif DataSet == 2
    disp('Running trained 1-bit-per-weight 20-10 Wide ResNet on the CIFAR10 test set')
    load('TrainedWideResNet20-10_CIFAR100_1-bit-per-weight','BinaryWeights','Moments','BNG','BNB','LayerWeights','Stride','Padding')
end
for i = 1:length(BinaryWeights)
    SingleBitWeights{i} = 2*single(BinaryWeights{i})-1; %convert boolean weights to -1 and 1
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step 3: run inference using the trained model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
FeatureMaps=single(TestImages);
for ConvLayer = 1:length(BinaryWeights)
    disp(['Currenly processing all images in layer: ' num2str(ConvLayer)])
    
    %BN--> ReLU --> convolution block
    FeatureMaps = vl_nnbnorm(FeatureMaps,BNG{ConvLayer},BNB{ConvLayer},'moments',Moments{ConvLayer});
    FeatureMaps = max(0,FeatureMaps);
    FeatureMaps = vl_nnconv(FeatureMaps,SingleBitWeights{ConvLayer}*LayerWeights(ConvLayer),[],'pad',Padding(ConvLayer),'stride',Stride(ConvLayer)); 
    
    if mod(ConvLayer,2)== 1
        %residual path
        if ConvLayer==1
            Residuals = FeatureMaps;
        else
            if size(Residuals,1) > size(FeatureMaps,1)
                %average poolwith stride 2 the residual path and double the channels using zero padding
                Residuals = vl_nnpool(Residuals,3,'method','avg','stride',2,'pad',1);
                Residuals = vl_nnconcat({Residuals,zeros(size(Residuals),'single')},3); 
            end
            Residuals = Residuals + FeatureMaps;
            FeatureMaps = Residuals;
        end
    end
end
%output block: batch norm and global average pooling
FeatureMaps = vl_nnbnorm(FeatureMaps,BNG{end},BNB{end},'moments',Moments{end});
FeatureMaps = sum(sum(FeatureMaps,1),2); %equivalent to global average pooling, since we ultimately use max
toc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step 4: quantify the error rate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,TopPrediction] = max(squeeze(FeatureMaps));
ErrorRate = 100*length(find(TopPrediction'~=labels_test))/length(labels_test)
