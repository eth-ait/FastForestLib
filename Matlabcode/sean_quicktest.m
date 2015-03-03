TreeNumber=1;
global size_figure; % image size of all the training figures
size_figure = [100,100];
global Depth_final; % depth of the whole tree
Depth_final = 16;
global Gestures_number; % number of gestures to be classified: 6 target gestures, 1 noisy gesture and 1 noise clasee 
Gestures_number = 2; 


load testT.mat; % load the decition forest
load testData.mat;
load forest.mat;
 pp = randperm(200);


for ii = 1:1
a = pp(ii);
seg_result_uint8 = data(:,:,a); % seg_result_uint8 for later manipulating
ground_truth = labels(:,:,a);
% b = (ground_truth == 1);
% c = (ground_truth == 2);
% d = (ground_truth == 3);
% [~,e] = max([sum(b(:)),sum(c(:)),sum(d(:))]);

%    cf_result_sj = zeros(size_figure(1),size_figure(2),Gestures_number);  
%    
% 
%    for tree =1 : TreeNumber
% %        T_temp = T{tree};
%           T_temp = T{tree};
% 
%       for i = 1 : size_figure(1)
%         for j = 1 : size_figure(2)
%             if seg_result_uint8(i,j)>0
%             
%                
%                p = decideTree_classification(seg_result_uint8,i,j,T_temp);
%                p = reshape(p,[1,1,Gestures_number]);
%                cf_result_sj(i,j,:) = cf_result_sj(i,j,:)+p;
%             end
%          end
%       end
%    end
%    cf_result_sj = cf_result_sj/TreeNumber; % normalize the classification result by the TreeNumber
% %    t1=toc;
% 
% 
%   
%    [~,gesture_mask] = max(cf_result_sj(:,:,1:Gestures_number),[],3); % It is the mask storing the output for gesture classification(8 gestures) at the final depth
%    gesture_mask = uint8(gesture_mask).*uint8(logical(seg_result_uint8)); % filter this mask by using segmentation binary mask
%    
%    % gesture 1 labeled as RED
% gesture1_mask = uint8(gesture_mask==1);
% gesture1_rgb = zeros(size_figure(1),size_figure(2),3);
% gesture1_rgb(:,:,1) = 255*gesture1_mask;
% 
% % gesture 2 labeled as GREEN
% gesture2_mask = uint8(gesture_mask==2);
% gesture2_rgb = zeros(size_figure(1),size_figure(2),3);
% gesture2_rgb(:,:,2) = 255*gesture2_mask;
% result_gesture = gesture1_rgb+gesture2_rgb;
% imshow(result_gesture);


cf_result_benj = zeros(size_figure(1),size_figure(2),Gestures_number);  
   

   for tree =1 : TreeNumber
%        T_temp = T{tree};
          T_temp = forest{tree};

      for i = 1 : size_figure(1)
        for j = 1 : size_figure(2)
            if seg_result_uint8(i,j)>0
            
               
               p = decideTree_classification_benj(seg_result_uint8,i,j,T_temp);
               p = reshape(p,[1,1,Gestures_number]);
               cf_result_benj(i,j,:) = cf_result_benj(i,j,:)+p;
            end
         end
      end
   end
   cf_result_benj = cf_result_benj/TreeNumber; % normalize the classification result by the TreeNumber
   t1=toc;


  
   [~,gesture_mask] = max(cf_result_benj(:,:,1:Gestures_number),[],3); % It is the mask storing the output for gesture classification(8 gestures) at the final depth
   gesture_mask = uint8(gesture_mask).*uint8(logical(seg_result_uint8)); % filter this mask by using segmentation binary mask
   
   % gesture 1 labeled as RED
gesture1_mask = uint8(gesture_mask==1);
gesture1_rgb = zeros(size_figure(1),size_figure(2),3);
gesture1_rgb(:,:,1) = 255*gesture1_mask;

% gesture 2 labeled as GREEN
gesture2_mask = uint8(gesture_mask==2);
gesture2_rgb = zeros(size_figure(1),size_figure(2),3);
gesture2_rgb(:,:,2) = 255*gesture2_mask;
result_gesture_benj = gesture1_rgb+gesture2_rgb;
figure(2);
imshow(result_gesture_benj);

%    b = (gesture_mask == 1);
%    c = (gesture_mask == 2);
%    d = (gesture_mask == 3);
%    [~,f] = max([sum(b(:)),sum(c(:)),sum(d(:))]);
%    Output = zeros(1,7);
%    
%    for gesture_index = 1:7
%       Output(gesture_index) = sum(gesture_mask(:) == gesture_index);
%    end
% [~,Outputlabel] = max(Output);
% % 
% Output_figure = uint8(zeros(240,100,3));
% if Outputlabel == 1
%    Output_figure(:,:,1) = 255;
% elseif Outputlabel == 2
%    Output_figure(:,:,2) = 255; 
% elseif Outputlabel == 3
%    Output_figure(:,:,3) = 255; 
% elseif Outputlabel == 4
%    Output_figure(:,:,1) = 255; 
%    Output_figure(:,:,2) = 255; 
% elseif Outputlabel == 5
%    Output_figure(:,:,1) = 255; 
%    Output_figure(:,:,3) = 255; 
% elseif Outputlabel == 6
%    Output_figure(:,:,2) = 255; 
%    Output_figure(:,:,3) = 255; 
% elseif Outputlabel == 7
%    Output_figure(:,:,1) = 255; 
%    Output_figure(:,:,2) = 155; 
% end


% if frame_index>=1 && frame_index<202
% Groudtruthlabel = 1;
% elseif frame_index>=203 && frame_index<495
% Groudtruthlabel = 2;
% elseif frame_index>=495 && frame_index<700
% Groudtruthlabel = 3;
% elseif frame_index>=700 && frame_index<910
% Groudtruthlabel = 4;
% elseif frame_index>=910 && frame_index<1144
% Groudtruthlabel = 5;
% elseif frame_index>=1144 && frame_index<1299
% Groudtruthlabel = 6;
% elseif frame_index>=1299 && frame_index<1300
% Groudtruthlabel = 7;
% end
% 
% if frame_index>=1 && frame_index<109
% Groudtruthlabel = 1;
% elseif frame_index>=109 && frame_index<208
% Groudtruthlabel = 2;
% elseif frame_index>=208 && frame_index<345
% Groudtruthlabel = 3;
% elseif frame_index>=345 && frame_index<459
% Groudtruthlabel = 4;
% elseif frame_index>=459 && frame_index<589
% Groudtruthlabel = 5;
% elseif frame_index>=589 && frame_index<677
% Groudtruthlabel = 6;
% elseif frame_index>=677 && frame_index<842
% Groudtruthlabel = 7;
% end
% 
% 
% Groudtruth_figure = uint8(zeros(240,160,3));
% if Groudtruthlabel == 1
%    Groudtruth_figure(:,:,1) = 255;
% elseif Groudtruthlabel == 2
%    Groudtruth_figure(:,:,2) = 255; 
% elseif Groudtruthlabel == 3
%    Groudtruth_figure(:,:,3) = 255; 
% elseif Groudtruthlabel == 4
%    Groudtruth_figure(:,:,1) = 255; 
%    Groudtruth_figure(:,:,2) = 255; 
% elseif Groudtruthlabel == 5
%    Groudtruth_figure(:,:,1) = 255; 
%    Groudtruth_figure(:,:,3) = 255; 
% elseif Groudtruthlabel == 6
%    Groudtruth_figure(:,:,2) = 255; 
%    Groudtruth_figure(:,:,3) = 255; 
% elseif Groudtruthlabel == 7
%    Groudtruth_figure(:,:,1) = 255; 
%    Groudtruth_figure(:,:,2) = 155; 
% end
%     


% % gesture 1 labeled as RED
% gesture1_mask = uint8(gesture_mask==1);
% gesture1_rgb = zeros(size_figure(1),size_figure(2),3);
% gesture1_rgb(:,:,1) = 255*gesture1_mask;
% 
% % gesture 2 labeled as GREEN
% gesture2_mask = uint8(gesture_mask==2);
% gesture2_rgb = zeros(size_figure(1),size_figure(2),3);
% gesture2_rgb(:,:,2) = 255*gesture2_mask;

% % gesture 3 labeled as BLUE
% gesture3_mask = uint8(gesture_mask==3);
% gesture3_rgb = zeros(size_figure(1),size_figure(2),3);
% gesture3_rgb(:,:,3) = 255*gesture3_mask;
% 
% % gesture 4 labeled as YELLOW
% gesture4_mask = uint8(gesture_mask==4);
% gesture4_rgb = zeros(size_figure(1),size_figure(2),3);
% gesture4_rgb(:,:,1) = 255*gesture4_mask;
% gesture4_rgb(:,:,2) = 255*gesture4_mask;
% gesture4_rgb(:,:,3) = 255*gesture4_mask;
% 
% gesture5_mask = uint8(1-seg_result_uint8);
% gesture5_rgb = zeros(size_figure(1),size_figure(2),3);
% gesture5_rgb(:,:,1) = 255*gesture5_mask;
% gesture5_rgb(:,:,2) = 255*gesture5_mask;
% gesture5_rgb(:,:,3) = 255*gesture5_mask;


% GT_mask = uint8(logical(seg_result_uint8));
% GT = zeros(size_figure(1),size_figure(2),3);
% GT(:,:,1) = 255*GT_mask;
% GT = GT+gesture5_rgb;

% 
% % gesture 5 labeled as PRUPLE
% gesture5_mask = uint8(gesture_mask==5);
% gesture5_rgb = zeros(size_figure(1),size_figure(2),3);
% gesture5_rgb(:,:,1) = 255*gesture5_mask;
% gesture5_rgb(:,:,3) = 255*gesture5_mask;
% 
% % gesture 6 labeled as Cyan
% gesture6_mask = uint8(gesture_mask==6);
% gesture6_rgb = zeros(size_figure(1),size_figure(2),3);
% gesture6_rgb(:,:,2) = 255*gesture6_mask;
% gesture6_rgb(:,:,3) = 255*gesture6_mask;
% 
% % gesture 7 labeled as Orange
% gesture7_mask = uint8(gesture_mask==7);
% gesture7_rgb = zeros(size_figure(1),size_figure(2),3);
% gesture7_rgb(:,:,1) = 255*gesture7_mask;
% gesture7_rgb(:,:,2) = 155*gesture7_mask;
% 
% 
% % noise labeled as WHITE
% gesture8_mask = uint8(gesture_mask==8);
% gesture8_rgb = zeros(size_figure(1),size_figure(2),3);
% gesture8_rgb(:,:,1) = 255*gesture8_mask;
% gesture8_rgb(:,:,2) = 255*gesture8_mask;
% gesture8_rgb(:,:,3) = 255*gesture8_mask;


% if frame_index >=10
% hand_xy_history = sum_part_mask_last10/10;
% mask_ha_history = hand_xy_history > 0.85;
% mask_bg_history = hand_xy_history < 0.05;
% hand_index_history = find(mask_ha_history == 1);
% bg_index_history = find(mask_bg_history == 1);
% end
% result_gesture = gesture1_rgb+gesture2_rgb+gesture3_rgb+gesture4_rgb+gesture5_rgb+gesture6_rgb; %final gesture colored image


% result_gesture = gesture1_rgb+gesture2_rgb+gesture3_rgb+gesture4_rgb+gesture5_rgb+gesture6_rgb+gesture7_rgb+gesture8_rgb;

% result_gesture = gesture1_rgb+gesture2_rgb;



% result_gesture = gesture1_rgb+gesture2_rgb+gesture3_rgb+gesture4_rgb+gesture5_rgb;

% result_gesture_GT = uint8(result_gesture);
% result_gesture(:,:,1) = result_gesture(:,:,1).*(seg_result_uint8);
% result_gesture(:,:,2) = result_gesture(:,:,2).*(seg_result_uint8);
% result_gesture(:,:,3) = result_gesture(:,:,3).*(seg_result_uint8);

% result_gesture(:,:,1) = result_gesture(:,:,1);
% result_gesture(:,:,2) = result_gesture(:,:,2);
% result_gesture(:,:,3) = result_gesture(:,:,3);
% result_gesture(:,:,1) = result_gesture(:,:,1);
% result_gesture(:,:,2) = result_gesture(:,:,2);
% result_gesture(:,:,3) = result_gesture(:,:,3);

% imshow(result_gesture);

% result_part_depth_switch = part_mask_depth_switch;
% result_part_depth_switch = uint8(result_part_depth_switch);
% result_part_depth_switch = result_part_depth_switch.*(1-seg_result_uint8);
% result_part_depth_switch = logical(result_part_depth_switch);
% figure(2);
% imshow(result_part_depth_switch);

% result_part_depth_final = part_mask_depth_final;
% result_part_depth_final = uint8(result_part_depth_final);
% result_part_depth_final = result_part_depth_final.*(1-seg_result_uint8);
% result_part_depth_final = logical(result_part_depth_final);
% figure(3);
% imshow(result_part_depth_final);

% demo(1:240,1:320,1) = (1-seg_result_uint8)*255;
% demo(1:240,1:320,2) = (1-seg_result_uint8)*255;
% demo(1:240,1:320,3) = (1-seg_result_uint8)*255;
% demo(241:480,1:320,1) = (1-seg_result_uint8)*255;
% demo(241:480,1:320,2) = (1-seg_result_uint8)*255;
% demo(241:480,1:320,3) = (1-seg_result_uint8)*255;
% demo(1:240,321:640,1) = result_gesture(:,:,1);
% demo(1:240,321:640,2) = result_gesture(:,:,2);
% demo(1:240,321:640,3) = result_gesture(:,:,3);
% demo(241:480,321:420,1) = Output_figure(:,:,1);
% demo(241:480,321:420,2) = Output_figure(:,:,2);
% demo(241:480,321:420,3) = Output_figure(:,:,3);
% demo(241:480,481:640,1) = Output_figure(:,:,1);
% demo(241:480,481:640,2) = Output_figure(:,:,2);
% demo(241:480,481:640,3) = Output_figure(:,:,3);
%  imshow(demo)
% if mod(ii,5) ==0
% result_gesture(result_gesture==0) = 255;
% imshow(result_gesture)
% imshow(GT)
% sjsj(ii) = (e-f);
% end

end