%% Train a Convolutional Neural Network for Regression
% This example shows how to fit a regression model using convolutional
% neural networks to predict the angles of rotation of handwritten digits.
%
% Convolutional neural networks (CNNs or ConvNets) are essential tools for
% deep learning, and are especially suited for analyzing image data. For
% example, you can use CNNs to classify images. To predict continuous data
% such as angles and distances, you can include a regression layer at the
% end of the network.
%
% The example constructs a convolutional neural network architecture,
% trains a network, and uses the trained network to predict angles of
% rotated, handwritten digits. These predictions are useful for optical
% character recognition.
%
% Optionally, you can use |imrotate| (Image Processing Toolbox(TM)) to
% rotate the images, and |boxplot| (Statistics and Machine Learning
% Toolbox(TM)) to create a residual box plot.

%% 
use_myDAGnet = 1;
do_generate_tbl = 0;

%% Load Training Data

% if do_generate_tbl
%     load('D:\matlab\DeepLearning\OpenPose\Realtime_Multi-Person_Pose_Estimation\training\dataset\COCO\mat\coco_kpt.mat')
% 
%     L = length(coco_kpt);
%     num_of_keypoints = 0;
%     for i = 1:L
%          img_name = sprintf('D:/matlab/DeepLearning/OpenPose/Realtime_Multi-Person_Pose_Estimation/training/dataset/COCO/images/train2014/COCO_train2014_%012d.jpg', coco_kpt(i).image_id);
%          save_address = sprintf('H:/DataSet/COCO/train2014_resize_368/COCO_train2014_%012d.jpg', coco_kpt(i).image_id);
%          annorect_keypoints_arry = cell2mat({coco_kpt(i).annorect.keypoints}');
%          annorect_num_keypoints = cell2mat({coco_kpt(i).annorect.num_keypoints}');
%          
%          k=0;clear annorect_keypoints_line;
%          for j=1:length( annorect_num_keypoints )
%              if(  annorect_num_keypoints(j)>0 )
%                  k=k+1;   
%                  annorect_keypoints_line(:,:, k) = reshape(annorect_keypoints_arry(j,:),3,[])';             
%              end
%          end
%          
%         
% 
%          if sum(annorect_num_keypoints)>0
%              num_of_keypoints = num_of_keypoints + 1;
%              
%              img = imread(img_name);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% %                 figure(1)
% %                 imshow(img); hold on
% %                 scatter(reshape(annorect_keypoints_line(:,1,:),1,[]),reshape(annorect_keypoints_line(:,2,:),1,[]))
% %                 hold off
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              [x,y,z] = size(img);
%              img = imresize(img, [368 368]);
%              imwrite(img, save_address);
%              annorect_keypoints_line(:,1,:) = annorect_keypoints_line(:,1,:).*(368/y);
%              annorect_keypoints_line(:,2,:) = annorect_keypoints_line(:,2,:).*(368/x); 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% %                 figure(2)
% %                 imshow(img); hold on
% %                 scatter(reshape(annorect_keypoints_line(:,1,:),1,[]),reshape(annorect_keypoints_line(:,2,:),1,[]))
% %                 hold off
% %                 pause(2)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              keypoints(num_of_keypoints,1) = {save_address};
%              for j=1:17
%                  clear raw 
%                  raw(:,:,:) = reshape(cell2mat({annorect_keypoints_line(j,1:2,:)}), 2,[]);
%                  
% %                  raw_kk = 0;
% %                  for kk=1:k   
% %                      if annorect_keypoints_line(j, 3, kk) ~=0
% %                            raw_kk = raw_kk + 1; 
% %                            raw(:,:,raw_kk) = cell2mat({annorect_keypoints_line(j,1:2,kk)});
% %                      end
% %                  end
% 
%                  if raw_kk>0
%                      keypoints(num_of_keypoints,j+1) = {reshape(raw, 2, [])'};
%                  end
%              end             
%          end         
%     end
% 
%     tbl_keypoints = cell2table(keypoints);%,...
%                     'VariableNames',{   'imageFilename' 'nose'	...
%                                         'left_eye' 'right_eye' ...
%                                         'left_ear' 'right_ear' ...
%                                         'left_shoulder' 'right_shoulder' ...
%                                         'left_elbow' 'right_elbow'...
%                                         'left_wrist' 'right_wrist' ...
%                                         'left_hip' 'right_hip' ...
%                                         'left_knee' 'right_knee' ...
%                                         'left_ankle' 'right_ankle' });
% else
%     load('tbl_keypoint_coco_kpt.mat.mat');
% end
L=length(keypoints_Boolean);
for i = 1:L
    keypoints_Boolean(i,3) = {single(cell2mat(keypoints_Boolean(i,3)))};
end
tbl_keypoints = cell2table(keypoints_Boolean(:,[1 3]));
 
%  imds = imageDatastore(img_paths);
% % [trainImages,~,trainAngles] = digitTrain4DArrayData;
% trainImages = imds;

%%
% % Display 20 random sample training digits using |imshow|.
% numTrainImages = size(trainImages,4);
% 
% figure
% idx = randperm(numTrainImages,20);
% for i = 1:numel(idx)
%     subplot(4,5,i)
%     
%     imshow(trainImages(:,:,:,idx(i)))
%     drawnow
% end

%% Create Network Layers
% To solve the regression problem, create the layers of the network and
% include a regression layer at the end of the network.

%%
% The first layer defines the size and type of the input data. The input
% images are 28-by-28-by-1. Create an image input layer of the same size as
% the training images.

%%
% The middle layers of the network define the core architecture of the
% network. Create a 2-D convolutional layer with 25 filters of size 12
% followed by a ReLU layer.

%%
% The final layers define the size and type of output data. For regression
% problems, a fully connected layer must precede the regression layer at
% the end of the network. Create a fully connected output layer of size 1
% and a regression layer.

%%
% Combine all the layers together in a |Layer| array.
if use_myDAGnet
%     run('D:\matlab\DeepLearning\test\TrainSemanticSegmentationNetworkExample\Builde_my_own_DAGnet.m')
    load('myDAGnet.mat')
    myDAGnet = removeLayers(myDAGnet, 'routput');
    myDAGnet = addLayers(myDAGnet,RegressionMAELayer_ly('Regression_Layer'));
%     myDAGnet = connectLayers(myDAGnet, 'Mconv7_stage6_L2', 'Regression_Layer');
    myDAGnet = connectLayers(myDAGnet, 'Mconv7_stage2_L2', 'Regression_Layer');
    layers = myDAGnet;
else
    layers = [ ...
        imageInputLayer([368 368 3])
        convolution2dLayer(12,25)
        reluLayer
        fullyConnectedLayer(2)
        regressionLayer];
end

%% Train Network
% Create the network training options. Set the initial learn rate to 0.001.
% To reduce training time, lower the value of |'MaxEpochs'|.
% options = trainingOptions('sgdm','InitialLearnRate',0.001, ...
%     'MaxEpochs',15);
options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate',1e-7,...%0.00001,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.8,...
    'LearnRateDropPeriod',1,...
    'MaxEpochs', 99, ...  
    'MiniBatchSize', 1, ...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 2, ...
    'ExecutionEnvironment','gpu',...
    'CheckpointPath','H:\DataSet\COCO\coco_kpt_pose_Mask_resize\checkpoint',...
    'Plots','training-progress');

%%
% Create the network using |trainNetwork|. This command uses a compatible
% GPU if available. Otherwise, |trainNetwork| uses the CPU. A
% CUDA(R)-enabled NVIDIA(R) GPU with compute capability 3.0 or higher is
% required for training on a GPU. Training can take a few minutes,
% especially when training on a CPU.
% net = trainNetwork(trainImages,trainAngles,layers,options)
% net = trainNetwork(trainImages,trainAngles,layers,options)
% [trainedNet,infrom] = trainNetwork(tbl_keypoints,net.Layers,options);     % %¼ÌÐøÑµÁ·

[trainedNet,infrom] = trainNetwork(tbl_keypoints,layers,options);

%%
% Examine the details of the network architecture contained in the |Layers|
% property of |net|.
net.Layers

%% Test Network
% Test the performance of the network by evaluating the prediction accuracy
% of held out test data.
%
% Load the digit test set.
[testImages,~,testAngles] = digitTest4DArrayData;

%%
% Use |predict| to predict the angles of rotation of the test images.
predictedTestAngles = predict(net,testImages);

%%
% *Evaluate Performance*
%
% Evaluate the performance of the model by calculating:
%%
% # The percentage of predictions within an acceptable error margin
% # The root-mean-square error (RMSE) of the predicted and actual angles of
% rotation
%
% Calculate the prediction error between the predicted and actual angles of
% rotation.
predictionError = testAngles - predictedTestAngles;

%%
% Calculate the number of predictions within an acceptable error margin
% from the true angles. Set the threshold to be 10 degrees. Calculate the
% percentage of predictions within this threshold.
thr = 10;
numCorrect = sum(abs(predictionError) < thr);
numTestImages = size(testImages,4);

accuracy = numCorrect/numTestImages

%%
% Use the root-mean-square error (RMSE) to measure the differences between
% the predicted and actual angles of rotation.
squares = predictionError.^2;
rmse = sqrt(mean(squares))

%%
% If the accuracy is too low, or the RMSE is too high, then try increasing
% the value of |'MaxEpochs'| in the call to |trainingOptions|.

%%
% *Display Box Plot of Residuals for Each Digit Class*
%
% Calculate the residuals.
residuals = testAngles - predictedTestAngles;

%%
% The |boxplot| function requires a matrix where each column corresponds to
% the residuals for each digit class.
%
% The test data groups images by digit classes 0&ndash;9 with 500 examples
% of each. Use |reshape| to group the residuals by digit class.
residualMatrix = reshape(residuals,500,10);

%%
% Each column of |residualMatrix| corresponds to the residuals of each
% digit. Create a residual box plot for each digit using |boxplot|
% (Statistics and Machine Learning Toolbox).
figure
boxplot(residualMatrix, ...
    'Labels',{'0','1','2','3','4','5','6','7','8','9'})

xlabel('Digit Class')
ylabel('Degrees Error')
title('Residuals')

%%
% The digit classes with highest accuracy have a mean close to zero and
% little variance.

%% Correct Digit Rotations
% You can use functions from Image Processing Toolbox to straighten the
% digits and display them together. Rotate 49 sample digits according to
% their predicted angles of rotation using |imrotate| (Image Processing
% Toolbox).
idx = randperm(numTestImages,49);
for i = 1:numel(idx)
    image = testImages(:,:,:,idx(i));
    predictedAngle = predictedTestAngles(idx(i));
    
    imagesRotated(:,:,:,i) = imrotate(image,predictedAngle,'bicubic','crop');
end

%%
% Display the original digits with their corrected rotations. You can use
% |montage| (Image Processing Toolbox) to display the digits together in a
% single image.
figure
subplot(1,2,1)
montage(testImages(:,:,:,idx))
title('Original')

subplot(1,2,2)
montage(imagesRotated)
title('Corrected')