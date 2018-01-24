%% ����openpose_pice Caffe�����
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
%% ����LayerGraph����,�����openpose_pice�����
myDAGnet = layerGraph(layers{1});
for i = 2:L
    myDAGnet = addLayers(myDAGnet, layers{i}(2:length(layers{i})));    
end
plot(myDAGnet)


%% ���concat��
for i = 1: length(concat_name)
    myDAGnet = addLayers(myDAGnet,depthConcatenationLayer(2,'Name',concat_name{i}));
end

%%
% ����1 2����ṹ
myDAGnet = connectLayers(myDAGnet, layers{1, 1}(1, length(layers{1, 1})-1).Name, layers{1, 2}(1,2).Name);
% 2��ֿ�����
for i = 1:length(concat_name)
    myDAGnet = connectLayers(myDAGnet, layers{1, 1}(1, length(layers{1, 1})-1).Name, [concat_name{i},'/in1']);
end
% ���ӵ�concat��
for i = 2:(length(concat_name)+1)
    myDAGnet = connectLayers(myDAGnet, layers{1, i}(1, length(layers{1, i})).Name, [concat_name{i-1},'/in2']);
end
% concat�����ӵ�
for i = 1:length(concat_name)
    myDAGnet = connectLayers(myDAGnet, concat_name{i}, layers{1, i+2}(1, 2).Name);
%     layers{1, i+2}(1, 2).NumChannels = layers{1, i+2}(1, 2).NumChannels - 38;
end

%���ھ���openpose������ṹ��
%% ��������
myDAGnet = removeLayers(myDAGnet, 'relu4_4_CPM');
myDAGnet = addLayers(myDAGnet,regressionLayer('Name', 'routput'));
myDAGnet = connectLayers(myDAGnet, layers{1, L}(1, length(layers{1, i})).Name, 'routput')
% myDAGnet = connectLayers(myDAGnet, 'Mconv7_stage2_L2', 'routput')
plot(myDAGnet)    


