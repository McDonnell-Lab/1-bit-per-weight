clear all

%dependencies: matlab, matlab parallel processing toolbox, matconvnet

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step 0. Set up matconvnet and GPU training
%        Note: we use only the following functions from matconvnet:
%           vl_nnbnorm, vl_nnconv, vl_nnpool, vl_nnsoftmaxloss, vl_nnsoftmax
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Setting up...')

%specifiy which GPU
gpuDevice(1);

%make sure we can see matconvnet
addpath('C:\Users\mcdonnmd\Documents\MATLAB\matconvnet-1.0-beta25\matconvnet-1.0-beta25\matlab')
vl_setupnn

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step 1. Set parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PatchSize = 18;
WidthFactor = 4;
UseSign = 1;
UseCutout = 1; 
DataSet = 'CIFAR10'; %currently setup for 'CIFAR10' and 'CIFAR100'
MaxEpochs = 254; %should be one of 2,6,14,30,62,126,254,510... for the warm restart method
ConfigID = 1; %arbitrary number to indicate your model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step 2. load and format the raw data and put into the right format for matconvnet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%download and load CIFAR-100 test data
[X,X_test,labels,labels_test,K_train,K_test,NumClasses] = LoadData(DataSet);

%create target one-hot matrix
Y = zeros(1,1,NumClasses,K_train,'single');
for i = 1:K_train
   Y(1,1,labels(i),i) = 1; 
end

toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step 3. load network architecture and algorithm training design parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LayerTypes,NumChannels,W,Stride,Padding,Training,DownLayers,BNChannels] = Setup_WideResNet(UseSign,...
                                                                           UseCutout,WidthFactor,PatchSize,NumClasses);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step 4. Initialise filters and setup memory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Filters = cell(length(W),1);
FilterUpdate = cell(length(W),1);

%create storage for batch normalisation parameters
NumBN = length(BNChannels);
BNG = cell(NumBN,1);
BNB = cell(NumBN,1);
BNG_u = cell(NumBN,1);
BNB_u = cell(NumBN,1);
for i = 1:NumBN
    BNB{i} = gpuArray.zeros(BNChannels(i),1,'single');
    BNG{i} = gpuArray.ones(BNChannels(i),1,'single');
    BNB_u{i} = gpuArray.zeros(BNChannels(i),1,'single');
    BNG_u{i} = gpuArray.zeros(BNChannels(i),1,'single');
end

%init filters using the method of Kaiming He
LayerWeights = gpuArray.zeros(length(Filters),1,'single');
for i = 1:length(Filters)
    LayerWeights(i) = sqrt(2/W(i)/W(i)/NumChannels(i));
    Filters{i} = gpuArray.randn(W(i),W(i),NumChannels(i),NumChannels(i+1),'single') * LayerWeights(i);
    FilterUpdate{i} = gpuArray.zeros(size(Filters{i}),'single');
end

%create memory for residual layers. Used for dealing with stride greater than 1
InputSize = size(X,1);
FeatureMapSizes = ceil([InputSize,InputSize/2,InputSize/4]);
if Training.ResNet
    RR{1} = gpuArray.zeros(FeatureMapSizes(2),FeatureMapSizes(2),NumChannels(DownLayers(1)+2),Training.BatchSize,'single');
    RR{2} = gpuArray.zeros(FeatureMapSizes(3),FeatureMapSizes(3),NumChannels(DownLayers(2)+2),Training.BatchSize,'single');
    Rdummy{1} = gpuArray.zeros(FeatureMapSizes(1),FeatureMapSizes(1),NumChannels(DownLayers(1)),Training.BatchSize,'single');
    Rdummy{2} = gpuArray.zeros(FeatureMapSizes(2),FeatureMapSizes(2),NumChannels(DownLayers(2)),Training.BatchSize,'single');
end

disp('done initialisizing')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step 5. Train the network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Training.DoPlot
    f1 = figure;hold on
end
OutputFileName = ['TrainedWRN_' DataSet];

IterationCount = 1;
CrossEntropyLoss = zeros(MaxEpochs*floor(K_train/Training.BatchSize),1);
for Epoch = 1:MaxEpochs
    
    tic
    
    %Randomly shuffle all training samples for this epoch. With SGD, this changes in each Epoch.
    Inds = randperm(K_train,K_train);
    
    %set up warm-restart learning schedule
    if Epoch  ==1 || Epoch == 3 || Epoch == 7 || Epoch == 15 || Epoch == 31 || Epoch == 63   || Epoch == 127
        Tmult = (Epoch+1);
        StartEpoch = Epoch;
        
        %seset the filter and BN memory for use with momentum
        clear FilterUpdate
        for i = 1:length(W)
            FilterUpdate{i} = gpuArray.zeros(size(Filters{i}),'single');
        end
        clear BNB_u BNG_u
        for i = 1:NumBN
            
            BNB_u{i} = gpuArray.zeros(BNChannels(i),1,'single');
            BNG_u{i} = gpuArray.zeros(BNChannels(i),1,'single');
        end
    end
    
    
    for j = 1:floor(K_train/Training.BatchSize)
        
        %warm-restart learning schedule - update on each batch
        Decay = 0.5+0.5*cos(pi*((Epoch-StartEpoch+j/floor(K_train/Training.BatchSize))/Tmult));
        Training.LearningRate = Training.FinalLearningRate+(Training.InitialLearningRate-Training.FinalLearningRate)*Decay;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %SGD step A: get training batch 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ThisInds = Inds((j-1)*Training.BatchSize+1:j*Training.BatchSize);
        B_forward = X(:,:,:,ThisInds);
        Y_batch = gpuArray(Y(:,:,:,ThisInds));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %SGD step B: augment (we do this on CPU, as it seems to be faster than GPU for augmentation)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        B_forward = AugmentData(B_forward,Training);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %SGD step B: forward prop through the net and get error for each sample in the minibatch
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        B_forward = single(gpuArray(B_forward));
        
        WeightLayerIndex = 1;
        BNcount=1;
        DownCount = 1;
        Residuals = [];
        for Layer = 2:length(LayerTypes)
           
            switch LayerTypes{Layer}
                case 'brw'
                    %store activations for backward prop
                    BP_BN{Layer} = B_forward; 
                    
                    %Batch norm, Relu, convolution
                    B_forward = vl_nnbnorm(B_forward,BNG{BNcount},BNB{BNcount},'Epsilon',1e-5);
                    B_forward = max(0,B_forward);
                    
                    %store activations for backward prop
                    BP_conv{Layer} = B_forward;
                    
                    %apply convolutional filters
                    if Training.UseSign
                        ConvWeights{WeightLayerIndex} = LayerWeights(WeightLayerIndex)*sign(Filters{WeightLayerIndex});
                    else
                        ConvWeights{WeightLayerIndex} =Filters{WeightLayerIndex};
                    end
                    B_forward = vl_nnconv(B_forward,ConvWeights{WeightLayerIndex},[],'pad',Padding(WeightLayerIndex),'stride',Stride(WeightLayerIndex));
                    
                    WeightLayerIndex = WeightLayerIndex + 1;
                    BNcount = BNcount + 1;
                case 'Residual_sum'
                    if isempty(Residuals)
                        Residuals = B_forward;
                    else
                        if size(Residuals,1) > size(B_forward,1)
                            %We have a downsampling skip path. Average pool the residual path with stride 2 and double the channels using zero-padding.
                            RR{DownCount}(:,:,1:size(RR{DownCount},3)/2,:) = vl_nnpool(Residuals,3,'method','avg','stride',2,'pad',1);
                            Residuals = RR{DownCount};
                            DownCount = DownCount + 1;
                        end
                        
                        %add the convolutional path to the residual path
                        Residuals = Residuals + B_forward;
                        B_forward = Residuals;
                    end
                case 'brg'
                    %store activations for backward prop
                    B_BP_BN_GAP = B_forward;
                    
                    %Batch norm and global average pooling
                    B_forward = vl_nnbnorm(B_forward,BNG{BNcount},BNB{BNcount},'Epsilon',1e-5);
                    B_forward = vl_nnpool(B_forward,size(B_forward,1),'method','avg');
            end
            
        end
        
        %calculate the loss for this batch
        CrossEntropyLoss(IterationCount) = gather(vl_nnsoftmaxloss(B_forward, labels(ThisInds)))/ Training.BatchSize;
        
        %start backward propagation
        DY = vl_nnsoftmax(B_forward)-Y_batch;

        WeightLayerIndex = length(Filters);
        BNcount = length(BNG);
        DownCount = length(DownLayers);
        DY_end = [];
        for Layer = length(LayerTypes):-1:2 %no need to use the input layer or the softmax layer
            switch LayerTypes{Layer}
                case 'brw'
                    %Backprop through conv layer. 
                    [DY,dFilters] = vl_nnconv(BP_conv{Layer},ConvWeights{WeightLayerIndex},[],DY,'pad',Padding(WeightLayerIndex),'stride',Stride(WeightLayerIndex));
                    
                    %update filter weights using SGD, momentum and weightdecay
                    FilterUpdate{WeightLayerIndex} =  Training.Momentum*FilterUpdate{WeightLayerIndex} - (dFilters*(Training.LearningRate/Training.BatchSize) + ...
                        (Training.LearningRate*Training.WeightDecay)*Filters{WeightLayerIndex});
                    Filters{WeightLayerIndex} = Filters{WeightLayerIndex} + FilterUpdate{WeightLayerIndex};

                     WeightLayerIndex = WeightLayerIndex - 1;
                    
                     %Backprop through ReLU layer
                     DY = (BP_conv{Layer}>0).*DY;
                     
                     %Backprop through BN layer
                     if BNcount==1
                         %input BN layer only
                         %update BN parameters using SGD and momentum
                         [~,dBNG,dBNB] = vl_nnbnorm(BP_BN{Layer},BNG{BNcount},BNB{BNcount},DY,'Epsilon',1e-5);
                         BNG_u{BNcount} = Training.Momentum*BNG_u{BNcount} - (Training.LearningRate/Training.BatchSize)*dBNG;
                         BNB_u{BNcount} = Training.Momentum*BNB_u{BNcount} - (Training.LearningRate/Training.BatchSize)*dBNB;
                         BNG{BNcount} = BNG{BNcount}  + BNG_u{BNcount};
                         BNB{BNcount} = BNB{BNcount}  + BNB_u{BNcount};
                     else
                         %Backprop through BN layer, without updating the BN parameters
                         [DY,~,~] = vl_nnbnorm(BP_BN{Layer},BNG{BNcount},BNB{BNcount},DY);
                     end
                     BNcount = BNcount - 1;
                case 'Residual_sum'
                    if ~isempty(DY_end)
                        if  size(DY_end,1) < size(DY,1)
                            %upsample if necessary
                            DY_end = DY_end(:,:,1:size(DY_end,3)/2,:);
                            DY_end = vl_nnpool(Rdummy{DownCount},3,DY_end,'method','avg','stride',2,'pad',1);
                            DownCount = DownCount-1;
                        end
                        DY = DY_end+DY;
                    end
                    DY_end = DY;
                case 'brg'
                    %Backprop through global average pooling layer
                    DY = repmat(DY,[FeatureMapSizes(end),FeatureMapSizes(end),1,1])/FeatureMapSizes(end)^2;

                    %Backprop through BN layer, without updating the BN parameters
                    [DY,~,~] = vl_nnbnorm(B_BP_BN_GAP,BNG{BNcount},BNB{BNcount},DY,'Epsilon',1e-5);
                    
                    BNcount = BNcount - 1;
            end
            
        end
   
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %for diagnostics, keep track of the loss and plot cross entropy loss (note: not the total loss function)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if mod(IterationCount,Training.DisplayIterationInterval)==0
            if Training.DoPlot
                %plot the smoothed logloss
                plot(conv(CrossEntropyLoss(1:IterationCount),ones(Training.PlotLossSmoothWindow,1)/Training.PlotLossSmoothWindow,'valid'),'r');drawnow
            end
            TimeSoFar = toc;
            disp(['Epoch = ' num2str(Epoch) '; Current learning rate = ' num2str(Training.LearningRate) '; time = ' num2str(TimeSoFar)])
        end
        IterationCount = IterationCount + 1;
        
    end
    toc
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Apply to the test data  after each epoch.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for WeightLayerIndex = 1:length(Filters)
        if Training.UseSign
            ConvWeights{WeightLayerIndex} = LayerWeights(WeightLayerIndex)*sign(Filters{WeightLayerIndex});
            BinaryWeights{WeightLayerIndex} = boolean(0.5*(sign(Filters{WeightLayerIndex})+1));
        else
            ConvWeights{WeightLayerIndex} =Filters{WeightLayerIndex};
        end
    end
    
    if Epoch  ==2 || Epoch == 6 || Epoch == 14 || Epoch == 30 || Epoch == 62 || Epoch == 126  || Epoch == 254
        NumTrainBatchesForMoments = K_train/Training.BatchSize; %maximum number
        [PercentCorrectTest(Epoch),~,Moments]=WideResNet_Test(ConvWeights,LayerTypes,Stride,Padding,BNG,BNB,RR,X,X_test,labels_test,Training,NumTrainBatchesForMoments);
        if Training.UseSign
            save([OutputFileName '_1_bit_weights_Width_' num2str(WidthFactor) '_' DataSet '_Config' num2str(ConfigID) '_Epoch_',num2str(Epoch),'.mat'],'BinaryWeights','LayerWeights',...
                                        'BNG','BNB','LayerTypes','W','NumChannels','Padding','Stride','Moments');
        else
             save([OutputFileName '_Width_' num2str(WidthFactor) '_' DataSet '_Config' num2str(ConfigID) '_Epoch_',num2str(Epoch),'.mat'],'Filters',...
                                        'BNG','BNB','LayerTypes','W','NumChannels','Padding','Stride','Moments');
        end
    else
        NumTrainBatchesForMoments = 10; %much faster for interim checking
        PercentCorrectTest(Epoch)=WideResNet_Test(ConvWeights,LayerTypes,Stride,Padding,BNG,BNB,RR,X,X_test,labels_test,Training,NumTrainBatchesForMoments);
    end
    
end



