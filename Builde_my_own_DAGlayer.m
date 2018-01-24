%% 导入openpose_pice Caffe网络层
Modle_Dir ='D:\matlab\DeepLearning\OpenPose\openpose_piece\modle\';
Modle_name = {  'pose_piece_vgg',...
                'pose_piece_CPML2',...
                'pose_piece_stage2L2',...
                'pose_piece_stage3L2',...
                'pose_piece_stage4L2',...
                'pose_piece_stage5L2',...
                'pose_piece_stage6L2'
                };
concat_name = { 'concat_stage2',...
                'concat_stage3',...
                'concat_stage4',...
                'concat_stage5',...
                'concat_stage6',...
                };      
            
L = length(Modle_name);
for i = 1: L
    protofile = [Modle_Dir,Modle_name{i},'.prototxt'];
    layers{i} = importCaffeLayers(protofile);    
end


% for i = 1: length(concat_name)
%     concat{i} = additionLayer(3,'Name',concat_name{i});
% end
%% 创建LayerGraph对象,并添加openpose_pice网络层
myDAGnet = layerGraph(layers{1});
for i = 2:L
    myDAGnet = addLayers(myDAGnet, layers{i}(2:length(layers{i})));    
end
plot(myDAGnet)


%% 添加concat层
for i = 1: length(concat_name)
    myDAGnet = addLayers(myDAGnet,depthConcatenationLayer(2,'Name',concat_name{i}));
end

%%
% 链接1 2网络结构
myDAGnet = connectLayers(myDAGnet, layers{1, 1}(1, length(layers{1, 1})-1).Name, layers{1, 2}(1,2).Name);
% 2层分开链接
for i = 1:length(concat_name)
    myDAGnet = connectLayers(myDAGnet, layers{1, 1}(1, length(layers{1, 1})-1).Name, [concat_name{i},'/in1']);
end
% 链接到concat层
for i = 2:(length(concat_name)+1)
    myDAGnet = connectLayers(myDAGnet, layers{1, i}(1, length(layers{1, i})).Name, [concat_name{i-1},'/in2']);
end
% concat层链接到
for i = 1:length(concat_name)
    myDAGnet = connectLayers(myDAGnet, concat_name{i}, layers{1, i+2}(1, 2).Name);
%     layers{1, i+2}(1, 2).NumChannels = layers{1, i+2}(1, 2).NumChannels - 38;
end

%现在就是openpose的网络结构了
%% 添加输出层
myDAGnet = removeLayers(myDAGnet, 'relu4_4_CPM');
myDAGnet = addLayers(myDAGnet,regressionLayer('Name', 'routput'));
myDAGnet = connectLayers(myDAGnet, layers{1, L}(1, length(layers{1, i})).Name, 'routput')
% myDAGnet = connectLayers(myDAGnet, 'Mconv7_stage2_L2', 'routput')
plot(myDAGnet)    


