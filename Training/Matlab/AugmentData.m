function X_batch = AugmentData(X_batch,Training)

if Training.Flips == 1
    %for each sample in the mini-batch, randomly flip left-right
    for jj = 1:Training.BatchSize
        if rand > 0.5
            X_batch(:,:,:,jj) = fliplr(X_batch(:,:,:,jj));
        end
    end
end

if Training.Crops == 1
    %randomly shift and crop, padding with random integers
    X_padded = uint8(round(255*rand(size(X_batch,1)+2*Training.ShiftMax,size(X_batch,2)+2*Training.ShiftMax,size(X_batch,3),size(X_batch,4),'single')));
    X_padded(1+Training.ShiftMax:size(X_batch,1)+Training.ShiftMax,1+Training.ShiftMax:size(X_batch,2)+Training.ShiftMax,:,:) = X_batch;
    X_out = zeros(size(X_batch),'uint8');
    for i = 1:size(X_batch,4)
        Inds = randperm(2*Training.ShiftMax+1,2)-1;
        X_out(:,:,:,i) = X_padded(1+Inds(1):size(X_batch,1)+Inds(1),1+Inds(2):size(X_batch,2)+Inds(2),:,i);
    end
    X_batch = X_out; 
end

if Training.Cutout == 1
    %apply cutout augmentation, using random integers for the pixels removed
    for jj = 1:Training.BatchSize
        Loc1 =  randi([2-Training.PatchSize,32],1); %2-Patchsize guarantees pixel 1 or larger is cutout
        Loc2 =  randi([2-Training.PatchSize,32],1);
        Patch = round(255*rand(length(max(1,Loc1):min(32,Loc1+Training.PatchSize-1)),length(max(1,Loc2):min(32,Loc2+Training.PatchSize-1)),3));
        X_batch(max(1,Loc1):min(32,Loc1+Training.PatchSize-1),max(1,Loc2):min(32,Loc2+Training.PatchSize-1),:,jj) = Patch;
    end
end
