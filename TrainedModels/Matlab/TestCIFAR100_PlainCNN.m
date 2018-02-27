clear all
 
%download and load CIFAR-100 test data
tic
websave('cifar-100-matlab.tar.gz','http://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz');
gunzip('cifar-100-matlab.tar.gz');untar('cifar-100-matlab.tar');
load('cifar-100-matlab\test.mat','data','fine_labels')
TestImages=permute(reshape(data,[size(data,1),32,32,3,]),[3,2,4,1]);
toc
 
%load the trained model
load('CIFAR100_20-4-Plain_644kB','BooleanSingleBitWeights','BN_Moments','BN_Scales','BN_Offsets','Stride','Padding','LayerScales')
 
%convert boolean weights to -1 and 1
for i = 1:size(BooleanSingleBitWeights,2)
    SingleBitWeights{i} = 2*single(BooleanSingleBitWeights{i})-1;
end
 
%do inference using the trained CNN
tic

%Batch Normalization applied to input images
FeatureMaps = vl_nnbnorm(single(TestImages),BN_Scales{1},BN_Offsets{1},'moments',BN_Moments{1});
for Layer = 1: size(BooleanSingleBitWeights,2)
    %ReLU:
    FeatureMaps = max(0,FeatureMaps);
    
    %Convolution:
    FeatureMaps = vl_nnconv(FeatureMaps,SingleBitWeights{Layer},[],'pad',Padding(Layer),'stride',Stride(Layer));   
    
    %Equation (1):
    FeatureMaps = LayerScales(Layer)*FeatureMaps;
    
    %Batch Normalization:
    FeatureMaps = vl_nnbnorm(FeatureMaps,BN_Scales{Layer+1},BN_Offsets{Layer+1},'moments',BN_Moments{Layer+1});
end
FeatureMaps = sum(sum(FeatureMaps,1),2); %global pooling
toc

%quantify the error rate:
[~,TopPrediction] = max(squeeze(FeatureMaps));
ErrorRate = 100*length(find(TopPrediction'~=single(fine_labels+1)))/length(fine_labels)
