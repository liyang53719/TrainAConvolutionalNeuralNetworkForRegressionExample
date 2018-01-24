close all;

%% 配置


% {'nose', 'neck', 'Rsho', 'Relb', 'Rwri', ... 
%  'Lsho', 'Lelb', 'Lwri', ...
%  'Rhip', 'Rkne', 'Rank', ...
%  'Lhip', 'Lkne', 'Lank', ...
%  'Leye', 'Reye', 'Lear', 'Rear', 'pt19'};

classes = [  "nose", "neck", ...
             "Rsho", "Relb", "Rwri", ... 
             "Lsho", "Lelb", "Lwri", ...
             "Rhip", "Rkne", "Rank", ...
             "Lhip", "Lkne", "Lank", ...
             "Leye", "Reye", "Lear", "Rear", "background"];

%% 获取摄像头
%% 
try
    camera = webcam;
catch
    clear camera
    camera = webcam;
end

keepRolling = 1;

im = snapshot(camera);
[x, y, z] = size(im);
net_size = net.Layers(1, 1).InputSize;
partcolor=colormap(prism);
figure(1)

%% 循环读取
while keepRolling
    im = snapshot(camera);
    im_orj = im; 
    im=imresize(im, [net_size(1) net_size(2)]);
%     act1 = activations(net,im,'conv5_4_CPM_L2'); implay((act1.*10000));
%     C = semanticseg(im, net);  
%     B = labeloverlay(im, C, 'Colormap', cmap, 'Transparency',0.4);

%     mean = 0.5;
%     img_out = double(im)/256;  
%     img_out = double(img_out) - mean;
%     img_out = permute(img_out, [2 1 3]);
%     img_out = img_out(:,:,[3 2 1]); % BGR for opencv training in caffe !!!!!

	res = predict(net,im);
% % %查看每个图层的输出    
%     act1 = res;
%     sz = size(act1);
%     act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
%     montage(mat2gray(act1),'Size',[4 5])

% %放大置信热点图 
    for j=1:19
       Big_res(:,:,j)  = imresize(res(:,:,j), [x y]);
    end

% %给热点图上色
    max_of_res = max(max(max(Big_res)));
    
    for j=1:18
       color_res(:,:,1,j)  = Big_res(:,:,j).*(partcolor(j,1));
       color_res(:,:,2,j)  = Big_res(:,:,j).*(partcolor(j,2));
       color_res(:,:,3,j)  = Big_res(:,:,j).*(partcolor(j,3));
    end
    
% %叠加热点图

    Labled_res = sum(color_res,4);
    Labled_res = Labled_res - min(min(min(Labled_res)));
    Labled_res = Labled_res*255/max(max(max(Labled_res)));
    Labled_res = uint8(Labled_res);
        
    Labled_img = Labled_res+ im_orj*0.5 ;
    
    imshow(Labled_img) 
%     hold on
%     pixelLabelColorbar(cmap, classes);
%     hold off
    drawnow
end
% 
% figure
% cell_keypoint = table2cell(tbl_keypoints);
% for i=1:100
%     img = imread(cell2mat(cell_keypoint(i,1)));    
%     imshow(img); hold on
% 
%     for j=2:18
%         a=(cell2mat(keypoints(i,j)));
%         scatter(a(:,1),a(:,2));
%         hold on
%     end
%     drawnow
%     hold off
%     pause(1)
% end
% 
% for i=1:100
%     coco_kpt(i).annorect.keypoints
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function pixelLabelColorbar(cmap, classNames)
% % Add a colorbar to the current axis. The colorbar is formatted
% % to display the class names with the color.
% 
% colormap(gca,cmap)
% 
% % Add colorbar to current figure.
% c = colorbar('peer', gca);
% 
% % Use class names for tick marks.
% c.TickLabels = classNames;
% numClasses = size(cmap,1);
% 
% % Center tick labels.
% c.Ticks = 1/(numClasses*2):1/numClasses:1;
% 
% % Remove tick mark.
% c.TickLength = 0;
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%