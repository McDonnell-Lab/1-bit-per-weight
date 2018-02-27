function [PercentCorrect_test,Y_predicted_test,MeanMoments] = WideResNet_Test(ConvWeights,LayerTypes,Stride,Padding,BNG,BNB,RR,X,X_test,labels_test,Training,NumTrainBatchesForMoments)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Test step 1. Run some training batches through the network to get the batch norm stats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K_train = size(X,4);
N = size(ConvWeights{end},4);

Inds = randperm(K_train,K_train);
NumBatches =NumTrainBatchesForMoments;
BatchSize = Training.BatchSize;
for ii = 1:NumBatches
    
    ThisInds = Inds((ii-1)*BatchSize+1:ii*BatchSize);
    X_batch = X(:,:,:,ThisInds);
    
    X_batch = AugmentData(X_batch,Training);
    
    [~,Moments] = ForwardProp(single(gpuArray(X_batch)),ConvWeights,LayerTypes,Stride,Padding,0,BNG,BNB,[],RR);
    
    %add the moments for each BN layer over all batches
    if ii== 1
        MeanMoments = Moments;
    else
        for j = 1:length(Moments)
            MeanMoments{j} = MeanMoments{j} + Moments{j};
        end
    end
end

%get the average batch norm moments
for j = 1:length(Moments)
    MeanMoments{j}(:,1) = MeanMoments{j}(:,1)/NumBatches;
    MeanMoments{j}(:,2) = MeanMoments{j}(:,2)/NumBatches;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Test step 2. Run the test data through the network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K_test = size(X_test,4);
TestBatchSize = Training.BatchSize;
Y_predicted_test  = gpuArray.zeros(N,K_test,'single');
for j = 1:floor(K_test/TestBatchSize)
    ThisInds = (j-1)*TestBatchSize+1:j*TestBatchSize;
    Y_predicted_test(:,ThisInds) = ForwardProp(gpuArray(X_test(:,:,:,ThisInds)),ConvWeights,LayerTypes,Stride,Padding,1,BNG,BNB,MeanMoments,RR);
end
ThisIndsEnd = 1+TestBatchSize*floor(K_test/TestBatchSize):K_test;
ThisInds(1:length(ThisIndsEnd)) = ThisIndsEnd;
Y_end = ForwardProp(gpuArray(X_test(:,:,:,ThisInds)),ConvWeights,LayerTypes,Stride,Padding,1,BNG,BNB,MeanMoments,RR);
Y_predicted_test(:,ThisIndsEnd) = Y_end(:,1:length(ThisIndsEnd));

%classify and evaluate:
[~,ClassificationID_test] = max(Y_predicted_test);
PercentCorrect_test = 100*(1-length(find(ClassificationID_test'~=labels_test))/K_test)
toc


function [B_forward,Moments] = ForwardProp(B_forward,Filters,LayerTypes,Stride,Padding,TestFlag,BNG,BNB,MeanMoments,RR)

Moments = cell(length(BNG),1);

WeightLayerIndex = 1;
BNcount=1;
DownCount = 1;
Residuals = [];
B_forward = single(B_forward);
for Layer = 2:length(LayerTypes)
    
    switch LayerTypes{Layer}
        case 'brw'
            if TestFlag
                [B_forward,~] = vl_nnbnorm(B_forward,BNG{BNcount},BNB{BNcount},'moments',MeanMoments{BNcount});
            else
                [B_forward,Moments{BNcount}] = vl_nnbnorm(B_forward,BNG{BNcount},BNB{BNcount},'Epsilon',1e-5);
            end
            B_forward = max(0,B_forward);
            
            B_forward = vl_nnconv(B_forward,(Filters{WeightLayerIndex}),[],'pad',Padding(WeightLayerIndex),'stride',Stride(WeightLayerIndex));
            WeightLayerIndex = WeightLayerIndex + 1;
            BNcount = BNcount + 1;
        case 'Residual_sum'
            if isempty(Residuals)
                Residuals = B_forward;
            else
                if size(Residuals,1) > size(B_forward,1)
                    RR{DownCount}(:,:,1:size(RR{DownCount},3)/2,:) = vl_nnpool(Residuals,3,'method','avg','stride',2,'pad',1);
                    Residuals = RR{DownCount};
                    DownCount = DownCount + 1;
                end
                Residuals = Residuals + B_forward;
                B_forward = Residuals;
            end
        case 'brg'
            if TestFlag
                [B_forward,~] = vl_nnbnorm(B_forward,BNG{BNcount},BNB{BNcount},'moments',MeanMoments{BNcount});
            else
                [B_forward,Moments{BNcount}] = vl_nnbnorm(B_forward,BNG{BNcount},BNB{BNcount},'Epsilon',1e-5);
            end
            B_forward = vl_nnpool(B_forward,size(B_forward,1),'method','avg');
    end
end







