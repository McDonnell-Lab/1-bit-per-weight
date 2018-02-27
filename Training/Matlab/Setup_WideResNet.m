function [LayerTypes,NumChannels,W,ConvStride,Padding,Training,DownLayers,BNChannels] = Setup_WideResNet(UseSign,UseCutout,WidthFactor,PatchSize,NumClasses)

%*************************%*************************%*************************
%20 layer network
FiltersEachBlockEachScale = WidthFactor*[16,32,64];

ResidualModule = {'brw','Residual_sum','brw'};
BlocksEachScale = 3;
LayerTypes = ['input' ...
               repmat(ResidualModule,1,BlocksEachScale+1)...
               repmat(ResidualModule,1,BlocksEachScale)...
               repmat(ResidualModule,1,BlocksEachScale)...
               'brg'];
NumChannels = [3,repmat(FiltersEachBlockEachScale(1),1,2*BlocksEachScale+1),repmat(FiltersEachBlockEachScale(2),1,2*BlocksEachScale),repmat(FiltersEachBlockEachScale(3),1,2*BlocksEachScale),NumClasses];
W = [3*ones(1,3*2*BlocksEachScale+1) 1];
ConvStride = [ones(1,BlocksEachScale*2+1) 2 ones(1,BlocksEachScale*2-1) 2 ones(1,BlocksEachScale*2-1) 1];
Padding = [ones(1,3*2*BlocksEachScale+1) 0];
DownLayers = find(ConvStride > 1);
%*************************%*************************%*************************

%training parameters
Training.BatchSize = 125;
Training.InitialLearningRate = 0.1;
Training.FinalLearningRate = 1e-5;
Training.Momentum = 0.9;
Training.WeightDecay = 0.0005;
Training.Flips = 1;
Training.Crops = 1;
Training.Cutout = UseCutout;
Training.UseSign = UseSign;
Training.ShiftMax = 4; %determines the padding on each side when using random crops
Training.PatchSize = PatchSize;
Training.DoPlot = 1;
Training.ResNet = 1;
Training.DisplayIterationInterval = 10;
Training.PlotLossSmoothWindow = 10;


%set up batch norm layers
NumLayers= length(LayerTypes);
NumBN = 0;
PrevChannels = 1;
for Layer = 1:NumLayers
    switch LayerTypes{Layer}
        case{'brw','brg'}
            NumBN = NumBN + 1;
            BNChannels(NumBN) = NumChannels(PrevChannels);
            PrevChannels = PrevChannels+1;
    end
end

NumberOfFilterWeightsEachLayer = NumChannels(2:length(W)+1).*NumChannels(1:length(W)).*(W.^2)
TotalLearntConvWeights = sum(NumberOfFilterWeightsEachLayer)