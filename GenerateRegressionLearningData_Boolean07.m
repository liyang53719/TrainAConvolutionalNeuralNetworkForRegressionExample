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
k = 0;  %生成训练集的长度
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

for i = 23611:L;%1146:L
    if mode == 1
        img_paths = sprintf('D:/matlab/DeepLearning/OpenPose/Realtime_Multi-Person_Pose_Estimation/training/dataset/COCO/images/train2014/COCO_train2014_%012d.jpg', coco_kpt(i).image_id);
    else
        img_paths = sprintf('D:/matlab/DeepLearning/OpenPose/Realtime_Multi-Person_Pose_Estimation/training/dataset/COCO/images/val2014/COCO_val2014_%012d.jpg', coco_kpt(i).image_id);
    end
%         save_address = sprintf('H:/DataSet/COCO/train2014_resize_368/COCO_train2014_%012d.jpg', coco_kpt(i).image_id);    
    oriImg = imread(img_paths);
%     rszImg_2x2 = imresize(oriImg, [368*2 368*2]);
    
    oriImg = imresize(oriImg, [368 368]);
    scale0 = 368/size(oriImg, 1);         
    img_out = double(oriImg)/256;  
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

    final_score = net.forward({img_out});
    res = cell2mat(final_score);
    res = permute(res(:,:,1:19), [2 1 3]);
    
    % 0.01
    for j = 1:18
        Boolean_Res_001(:,:,j) = res(:,:,j)>(0.01*max(max(max(res))));
    end
    Boolean_Res_001(:,:,19) = res(:,:,19)>((1-0.01)*max(max(max(res))));
    
    % 0.03
    for j = 1:18
        Boolean_Res_003(:,:,j) = res(:,:,j)>(0.03*max(max(max(res))));
    end
    Boolean_Res_003(:,:,19) = res(:,:,19)>((1-0.03)*max(max(max(res))));
    
    % 0.06
    for j = 1:18
        Boolean_Res_006(:,:,j) = res(:,:,j)>(0.06*max(max(max(res))));
    end
    Boolean_Res_006(:,:,19) = res(:,:,19)>((1-0.06)*max(max(max(res))));
    
    % 0.1
    for j = 1:18
        Boolean_Res_010(:,:,j) = res(:,:,j)>(0.1*max(max(max(res))));
    end
    Boolean_Res_010(:,:,19) = res(:,:,19)>((1-0.1)*max(max(max(res))));
    
    % 0.3
    for j = 1:18
        Boolean_Res_030(:,:,j) = res(:,:,j)>(0.3*max(max(max(res))));
    end
    Boolean_Res_030(:,:,19) = res(:,:,19)>((1-0.3)*max(max(max(res))));
    
    % 0.6
    for j = 1:18
        Boolean_Res_060(:,:,j) = res(:,:,j)>(0.6*max(max(max(res))));
    end
    Boolean_Res_060(:,:,19) = res(:,:,19)>((1-0.6)*max(max(max(res))));
    
    
    save_this = 0;
    for j=1:18
        if( sum(sum(Boolean_Res_001(:,:,j))) > 25 )
            save_this = 1;
            break;
        end
    end
    
    if (save_this == 1)
        k = k+1;
        save_address = sprintf('H:/DataSet/COCO/train2014_resize_368_Boolean001/COCO_train2014_%012d.jpg', coco_kpt(i).image_id);
        imwrite(oriImg, save_address);  

        keypoints_Boolean(k,1) = {save_address};
        keypoints_Boolean(k,2) = {Boolean_Res_001(:,:,1:19)};
        keypoints_Boolean(k,3) = {Boolean_Res_003(:,:,1:19)};
        keypoints_Boolean(k,4) = {Boolean_Res_006(:,:,1:19)};
        keypoints_Boolean(k,5) = {Boolean_Res_010(:,:,1:19)};
        keypoints_Boolean(k,6) = {Boolean_Res_030(:,:,1:19)};
        keypoints_Boolean(k,7) = {Boolean_Res_060(:,:,1:19)};
    end
   
    clear res
    clear oriImg
    clear final_score
    clc
    sprintf('%d/%d...%d',i,L,k)
    
end


