close all;

addpath('D:/matlab/DeepLearning/OpenPose/Realtime_Multi-Person_Pose_Estimation/testing'); 
addpath('D:/matlab/DeepLearning/OpenPose/Realtime_Multi-Person_Pose_Estimation/testing/src'); 
addpath('D:/matlab/DeepLearning/OpenPose/Realtime_Multi-Person_Pose_Estimation/testing/util');
addpath('D:/matlab/DeepLearning/OpenPose/Realtime_Multi-Person_Pose_Estimation/testing/util/ojwoodford-export_fig-5735e6d/');

% For MPI, mode = 2. For COCO, mode = 1.
mode = 1;
param = config(mode);
model = param.model(param.modelID);
net = caffe.Net(model.deployFile, model.caffemodel, 'test');

mode = 1;
mean = 0.5;
twoLevel = 1;
test_model = 'pose';
% test_model = 'face';
% test_model = 'hand';

%% input Caffe network by matcaffe
switch test_model
    case 'pose'
        caffemodel = 'models\pose\coco\pose_iter_440000.caffemodel';
        deployFile = 'models\pose\coco\pose_deploy_linevec_fix.prototxt';
        net = caffe.Net(deployFile, caffemodel, 'test');
    case 'face'        
        caffemodel = 'models\face\pose_iter_116000.caffemodel';
        deployFile = 'models\face\pose_deploy_fix.prototxt';
        net = caffe.Net(deployFile, caffemodel, 'test');        
    case 'hand'
        caffemodel = 'models\hand\pose_iter_102000.caffemodel';
        deployFile = 'models\hand\pose_deploy_fix.prototxt';
        net = caffe.Net(deployFile, caffemodel, 'test');
end

% {'nose', 'neck', 'Rsho', 'Relb', 'Rwri', ... 
%  'Lsho', 'Lelb', 'Lwri', ...
%  'Rhip', 'Rkne', 'Rank', ...
%  'Lhip', 'Lkne', 'Lank', ...
%  'Leye', 'Reye', 'Lear', 'Rear', 'pt19'};
%%
% oriImg = imread('./sample_image/timg.jpg');
    
if mode == 1 
    load('D:\matlab\DeepLearning\OpenPose\Realtime_Multi-Person_Pose_Estimation\training\dataset\COCO\mat\coco_kpt.mat');
else
    load('D:\matlab\DeepLearning\OpenPose\Realtime_Multi-Person_Pose_Estimation\training\dataset\COCO\mat\coco_val.mat');
    coco_kpt = coco_val;
end

L = length(coco_kpt);

for i = 24242:L;%1146:L
    if mode == 1
        img_paths = sprintf('D:/matlab/DeepLearning/OpenPose/Realtime_Multi-Person_Pose_Estimation/training/dataset/COCO/images/train2014/COCO_train2014_%012d.jpg', coco_kpt(i).image_id);
    else
        img_paths = sprintf('D:/matlab/DeepLearning/OpenPose/Realtime_Multi-Person_Pose_Estimation/training/dataset/COCO/images/val2014/COCO_val2014_%012d.jpg', coco_kpt(i).image_id);
    end
        save_address = sprintf('H:/DataSet/COCO/train2014_resize_368/COCO_train2014_%012d.jpg', coco_kpt(i).image_id);
    oriImg = imread(img_paths);
%     oriImg = imresize(oriImg,[368 368]);    
    scale0 = 368/size(oriImg, 1); 
    
    img_out = imresize(oriImg,[368 368]); 
    imwrite(img_out, save_address);  
    img_out = double(img_out)/256;  
    img_out = double(img_out) - mean;
    try
        img_out = permute(img_out, [2 1 3]);
        img_out = img_out(:,:,[3 2 1]);
    catch
        a(:,:,1) = img_out;
        a(:,:,2) = img_out;
        a(:,:,3) = img_out;
        img_out = a;
        img_out = permute(img_out, [2 1 3]);
        img_out = img_out(:,:,[3 2 1]);
    end
    
%     [final_score, ~] = applyModel(oriImg, param, net, scale0, 1, 1, 0, twoLevel);
    final_score = net.forward({img_out});
    res = cell2mat(final_score);
    res = permute(res(:,:,1:19), [2 1 3]);
%     for j =1:19
%         res(:,:,j) = imresize(final_score(:,:,j),[46 46]);
%     end
    
%     a=cell2mat(res);
%     b=a(:,:,1:19);
    keypoints(i,1) = {save_address};
    keypoints(i,2) = {res(:,:,1:19)};
%     [x,y,~] = size(oriImg);
    

%     if mode == 1
%         img_name = sprintf('../training/dataset/COCO/pose_regression2014/train2014_pose_regression_%012d.png', coco_kpt(i).image_id);
%         imwrite(sm,img_name);   
%     else
%         img_name = sprintf('../training/dataset/COCO/pose_regression2014/val2014_pose_regression_%012d.png', coco_kpt(i).image_id);
%         imwrite(sm,img_name);            
%     end
    clear res
    clear oriImg
    clear final_score
    clc
    sprintf('%d/%d',i,L)
    
end


