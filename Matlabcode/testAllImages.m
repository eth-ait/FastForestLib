TreeNumber=1;
global size_figure; % image size of all the training figures
size_figure = [100,100];
global Depth_final; % depth of the whole tree
Depth_final = 10;
global Gestures_number; % number of gestures to be classified: 6 target gestures, 1 noisy gesture and 1 noise clasee 
Gestures_number = 2; 
% global resized_factor; % downsample factor
% resized_factor = 1/4;

% load ../data/trialData.mat;
load ../data/testData_corrected.mat;
NumberOfImages = size(data, 3);
% NumberOfImages = 200;

GT_label = zeros(NumberOfImages,1);
Pre_pc = zeros(Gestures_number,Gestures_number);
Pre_label = zeros(Gestures_number,Gestures_number);
% Pre_label = zeros(NumberOfImages,1);


%load ../forests/matlab_forest.mat; % load the decition forest
%for ii = 1:NumberOfImages
%
%    display(['Testing image ', int2str(ii), ' out of ', int2str(NumberOfImages)]);
%
%seg_result_uint8 = data(:,:,ii); % seg_result_uint8 for later manipulating
%ground_truth = labels(:,:,ii);
%ground_truth_label = max(ground_truth(:));
%GT_label(ii) = ground_truth_label;
%
%
% cf_result = zeros(size_figure(1),size_figure(2),Gestures_number);  
%   
%
%   
%   %% pass the canditates hand pixels into the Random Forest
%   tic;
%   for tree =1 : TreeNumber
%%        T_temp = T{tree};
%          T_temp = forest{tree};
%
%      for i = 1 : size_figure(1)
%        for j = 1 : size_figure(2)
%            if seg_result_uint8(i,j)>0
%            
%               
%               p = decideTree_classification_benj(seg_result_uint8,i,j,T_temp);
%%                p = p/sum(p(:));
%               p = reshape(p,[1,1,Gestures_number]);
%               cf_result(i,j,:) = cf_result(i,j,:)+p;
%            end
%         end
%      end
%   end
%   cf_result = cf_result/TreeNumber; % normalize the classification result by the TreeNumber
%   t1=toc;
%
%
%  
%   [~,gesture_mask] = max(cf_result(:,:,1:Gestures_number),[],3); % It is the mask storing the output for gesture classification(8 gestures) at the final depth
%   gesture_mask = uint8(gesture_mask).*uint8(logical(seg_result_uint8));% filter this mask by using segmentation binary mask
%   
%   for gesture_index = 1:Gestures_number
%       tem = (gesture_mask == gesture_index);
%       tem_track(gesture_index) = sum(tem(:));
%       Pre_pc(ground_truth_label,gesture_index) = Pre_pc(ground_truth_label,gesture_index)+sum(tem(:));
%   end
       
   
% [~,f] =  max(tem_track);
%
%  Pre_label(ground_truth_label,f) = Pre_label(ground_truth_label,f) + 1;
%  
%  
%end




%load ../forests/testT_10_10_10_300_300_jie.mat;
load ../forests/trialData/matlab_forest4.mat;
for ii = 1:NumberOfImages
    
    display(['Testing image ', int2str(ii), ' out of ', int2str(NumberOfImages)]);

seg_result_uint8 = data(:,:,ii); % seg_result_uint8 for later manipulating
ground_truth = labels(:,:,ii);
ground_truth_label = max(ground_truth(:));
GT_label(ii) = ground_truth_label;

 cf_result = zeros(size_figure(1),size_figure(2),Gestures_number);  
   

   
   %% pass the canditates hand pixels into the Random Forest
   tic;
   for tree =1 : TreeNumber
%        T_temp = T{tree};
          T_temp = T{tree};

      for i = 1 : size_figure(1)
%           display(['Testing row ', int2str(i), ' out of ', int2str(size_figure(1))]);
        for j = 1 : size_figure(2)
%             display(['Testing column ', int2str(j), ' out of ', int2str(size_figure(2))]);
            if seg_result_uint8(i,j)>0
            
               
               p = decideTree_classification(seg_result_uint8,i,j,T_temp);
%                p = p/sum(p(:));
               p = reshape(p,[1,1,Gestures_number]);
               cf_result(i,j,:) = cf_result(i,j,:)+p;
            end
         end
      end
   end
   cf_result = cf_result/TreeNumber; % normalize the classification result by the TreeNumber
   t1=toc;


  
   [~,gesture_label_mask] = max(cf_result(:,:,1:Gestures_number),[],3); % It is the mask storing the output for gesture classification(8 gestures) at the final depth
   gesture_label_mask = uint8(gesture_label_mask).*uint8(logical(seg_result_uint8));% filter this mask by using segmentation binary mask
   
   for gesture_index = 1:Gestures_number
       tem = (gesture_label_mask == gesture_index);
       tem_track(gesture_index) = sum(tem(:));
       Pre_pc(ground_truth_label,gesture_index) = Pre_pc(ground_truth_label,gesture_index)+sum(tem(:));
   end
     
 
 [~,f] =  max(tem_track);

  Pre_label(ground_truth_label,f) = Pre_label(ground_truth_label,f) + 1;
  
  
end

Pre_pc ./ repmat(sum(Pre_pc, 2), [1,2])







% for ii = 1:Gestures_number
%     tem_label = GT_label(GT_label == ii);
%     tem_predict = Pre_label(GT_label == ii);
%     tem_predict = tem_predict';
%     for iii = 1:Gestures_number
%     Output(ii,iii) = sum((tem_predict==iii))/size(tem_label,1);
%     end
% end
% average_accuracy = sum((Pre_label==GT_label))/NumberOfImages; 
