%% settings
clear all;
close all;
% load('T');
load('../data/trialData.mat');

%trainingpicture = 9126;
global size_figure;
size_figure = [100,100];
global Depth;
Depth = 10;% maximal depth of decision tree
global gesture_number;
gesture_number = 2;
global Splits;
Splits=10; % number of candidate thresholds at each node
global Taus;
Taus = 10;
global MinNode;
MinNode=100; % minimal size of a non-leaf node
TreeNumber=1;
global Picturenumber_pertree;
Picturenumber_pertree =1000;

global low_bound;
low_bound= 3;
global high_bound;
high_bound= 15;



% grouping = random_grouping(TreeNumber,trainingpicture/TreeNumber);
% grouping = random_grouping(TreeNumber,Picturenumber_pertree,trainingpicture);


%   matlabpool('local', 4);




 
 global I_current;
 I_current = data;
%  T=struct('move',zeros(2^(Depth-1)-1,4),...
%     'leaf',zeros(2^(Depth-1),2),...
%     'leaf_prob',zeros(2^(Depth-1),1),...
%     'depth',Depth);
 %% training
%  boundvector = linspace(lowbound, highbound , TreeNumber+1);
%  load trainingdata1-50_dense.mat;
%  load trainingdata1688_1500_group1;

 tic;
 
 for i = 1 : TreeNumber
     T_temp = single(zeros(2^(Depth)-1, 6+gesture_number));
%      load (strcat('trainingdata9126_8000_group', num2str(i),'_resized')) 
%      T{i,1}=create01Tree(I,T,labels,1);
     T{i,1}=buildtree_new_classification(T_temp,labels,1);
%  T=buildtree_new(I,T,labels,1);
     
  end

  t1=toc;
 
% save('../forests/matlab_forest2.mat');
save('../forests/trialData/matlab_forest4.mat','T','t1');
%    matlabpool close;
   
