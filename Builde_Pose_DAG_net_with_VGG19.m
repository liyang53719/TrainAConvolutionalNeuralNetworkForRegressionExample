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
    datafile  = [Modle_Dir,Modle_name{i},'.caffemodel'];
    layers{i} = importCaffeNetwork(protofile,datafile);    
end

%% ����LayerGraph����,�����openpose�����
Pose_DAG_net = layerGraph(layers{1}.Layers(1));

%% ����VGG19Ԥѵ������,����ǰʮ����ӵ��ҵ�����
net_vgg19 = vgg19;
Pose_DAG_net = addLayers(Pose_DAG_net, net_vgg19.Layers(2:24));

%% �������openpose CPM�����
Pose_DAG_net = addLayers(Pose_DAG_net, layers{1, 1}.Layers(25:28));

%% ��������Щ�����
Pose_DAG_net = connectLayers(Pose_DAG_net, 'input', 'conv1_1');
Pose_DAG_net = connectLayers(Pose_DAG_net, 'relu4_2', 'conv4_3_CPM');
plot(Pose_DAG_net)

%% �������openposeʣ��������
for i = 2:L
    Pose_DAG_net = addLayers(Pose_DAG_net, layers{i}.Layers(2:length(layers{i}.Layers)));    
end
plot(Pose_DAG_net)

%% ���concat��
for i = 1: length(concat_name)
    Pose_DAG_net = addLayers(Pose_DAG_net,depthConcatenationLayer(2,'Name',concat_name{i}));
end
plot(Pose_DAG_net)

%% ���������
% ����1 2����ṹ
Pose_DAG_net = connectLayers(Pose_DAG_net, layers{1, 1}.Layers(length(layers{1, 1}.Layers)-1).Name, layers{1, 2}.Layers(2).Name);
plot(Pose_DAG_net)
% 2��ֿ�����
for i = 1:length(concat_name)
    Pose_DAG_net = connectLayers(Pose_DAG_net, layers{1, 1}.Layers(length(layers{1, 1}.Layers)-1).Name, [concat_name{i},'/in1']);
end
plot(Pose_DAG_net)
% ���ӵ�concat��
for i = 2:(length(concat_name)+1)
    Pose_DAG_net = connectLayers(Pose_DAG_net, layers{1, i}.Layers(length(layers{1, i}.Layers)).Name, [concat_name{i-1},'/in2']);
end
plot(Pose_DAG_net)
% concat�����ӵ�
for i = 1:length(concat_name)
    Pose_DAG_net = connectLayers(Pose_DAG_net, concat_name{i}, layers{1, i+2}.Layers(2).Name);
end
plot(Pose_DAG_net)

%% ���ھ���openpose������ṹ��
%% ɾ��û�õ��Ĳ�
Pose_DAG_net = removeLayers(Pose_DAG_net, 'relu4_4_CPM');
plot(Pose_DAG_net)

%% ��������
Pose_DAG_net = addLayers(Pose_DAG_net,regressionLayer('Name', 'routput'));
Pose_DAG_net = connectLayers(Pose_DAG_net, layers{1, L}.Layers(length(layers{1, i}.Layers)).Name, 'routput');
% myDAGnet = connectLayers(myDAGnet, 'Mconv7_stage2_L2', 'routput')
plot(Pose_DAG_net)  
clear concat_name datafile i L layers Modle_Dir Modle_name net_vgg19 protofile


