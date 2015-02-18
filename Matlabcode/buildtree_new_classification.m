function T=buildtree_new_classification(T,label,k)
% I: training figures
% k: current working node
% T: current decision tree
% X: N*1 training data matrix, each row is one sample
% X_u1: N*1 training data matrix, each row is one sample's u1 index
% X_u2: N*1 training data matrix, each row is one sample's u2 index
% label: N*1 labels of training data, each entry takes value 0 or 1, 0 is
% the 'other' part, while 1 stands for the finger tip
% Depth: maximal depth of decision tree
% Splits: number of candidate jumps at each node
% MinNode: minimal size of a non-leaf node


global Depth;
global Splits;
global MinNode;
global low_bound;
global high_bound;
global Picturenumber_pertree;
global gesture_number;
global I_current;

global size_figure;

%% leaf node







[d ,~]=index2depth(k);

if d==Depth
    T(k,1) = 1;
   
    label(label == 0) = [];
    
    if size(label,2) > 0
    
     T(k,7:6+gesture_number) =  histc(label,1:gesture_number);
     
    else
    T(k,7:6+gesture_number) = NaN;    
    end
   
      
    return
end




tem_label = label;
tem_label(tem_label == 0) = [];
point_number = size(tem_label,2);

    
    

 if point_number<=MinNode
    T(k,2:5) = [0,0,0,0];
    T(k,7:6+gesture_number) =  histc(tem_label,1:gesture_number);
    T=buildtree_new_classification(T,label,left_child(k));
    T=buildtree_new_classification(T,[],right_child(k));
    return;
 else
   T(k,7:6+gesture_number) =  histc(tem_label,1:gesture_number);  
    
end
%% non-leaf node


[label_size,current_entropy]=getEntropy(label,gesture_number);


entropyDecreases = zeros(Splits,6);

    parfor split_index = 1:Splits
        
         [ w1, w2 ] = randomgenerator_offset(low_bound,high_bound);
         
        [ v1, v2 ] = randomgenerator_offset(low_bound,high_bound);
        entropyDecrease_temp = zeros(4,1);
         
         
         I_w1w2 = uint8(zeros(size_figure(1),size_figure(2),Picturenumber_pertree));
        
         
         if w1>=0
           if w2>=0
             I_w1w2(w1+1:size_figure(1),1:size_figure(2)-w2,:) = I_current(1:size_figure(1)-w1,w2+1:size_figure(2),:);
             
           else
             I_w1w2(w1+1:size_figure(1),abs(w2)+1:size_figure(2),:) = I_current(1:size_figure(1)-w1,1:size_figure(2)-abs(w2),:);
             
           end
          elseif w1<0
           if w2>=0
             I_w1w2(1:size_figure(1)+w1,1:size_figure(2)-w2,:) = I_current(-w1+1:size_figure(1),w2+1:size_figure(2),:);
             
           else
             I_w1w2(1:size_figure(1)+w1,abs(w2)+1:size_figure(2),:) = I_current(-w1+1:size_figure(1),1:size_figure(2)-abs(w2),:);
           end
         
         end
         
         I_v1v2 = uint8(zeros(size_figure(1),size_figure(2),Picturenumber_pertree));
        
         if v1>=0
           if v2>=0
             I_v1v2(v1+1:size_figure(1),1:size_figure(2)-v2,:) = I_current(1:size_figure(1)-v1,v2+1:size_figure(2),:);
             
           else
             I_v1v2(v1+1:size_figure(1),abs(v2)+1:size_figure(2),:) = I_current(1:size_figure(1)-v1,1:size_figure(2)-abs(v2),:);
             
           end
          elseif v1<0
           if v2>=0
             I_v1v2(1:size_figure(1)+v1,1:size_figure(2)-v2,:) = I_current(-v1+1:size_figure(1),v2+1:size_figure(2),:);
             
           else
             I_v1v2(1:size_figure(1)+v1,abs(v2)+1:size_figure(2),:) = I_current(-v1+1:size_figure(1),1:size_figure(2)-abs(v2),:);
           end
         
         end

        
        
        
        

           
        
            
            
           
            %% calculate entropyDecreases for 4 different split criteria                                               
            feature_label = uint8(I_w1w2.*I_v1v2);           
            left_label = feature_label.*label; % label matrix for the pixels going to left
            [left_size,left_entropy]= getEntropy(left_label,gesture_number) ;            
            right_label = (1-feature_label).*label;  % label matrix for the pixels going to left
            [right_size,right_entropy] = getEntropy(right_label,gesture_number) ;            
             % calculate the current entropyDecrease
            entropyDecrease_temp(1) = current_entropy-left_entropy*left_size/label_size-right_entropy*right_size/label_size;
       
        
            feature_label = uint8(I_w1w2.*(1-I_v1v2));            
            left_label = feature_label.*label; % label matrix for the pixels going to left
            [left_size,left_entropy]= getEntropy(left_label,gesture_number) ;            
            right_label = (1-feature_label).*label;  % label matrix for the pixels going to left
            [right_size,right_entropy] = getEntropy(right_label,gesture_number) ;            
             % calculate the current entropyDecrease
            entropyDecrease_temp(2) = current_entropy-left_entropy*left_size/label_size-right_entropy*right_size/label_size;
       
        
            feature_label = uint8((1-I_w1w2).*I_v1v2);            
            left_label = feature_label.*label; % label matrix for the pixels going to left
            [left_size,left_entropy]= getEntropy(left_label,gesture_number) ;            
            right_label = (1-feature_label).*label;  % label matrix for the pixels going to left
            [right_size,right_entropy] = getEntropy(right_label,gesture_number) ;       
             % calculate the current entropyDecrease
            entropyDecrease_temp(3) = current_entropy-left_entropy*left_size/label_size-right_entropy*right_size/label_size;
       
            feature_label = uint8((1-I_w1w2).*(1-I_v1v2));            
            left_label = feature_label.*label; % label matrix for the pixels going to left
            [left_size,left_entropy]= getEntropy(left_label,gesture_number) ;            
            right_label = (1-feature_label).*label;  % label matrix for the pixels going to left
            [right_size,right_entropy] = getEntropy(right_label,gesture_number) ;               
             % calculate the current entropyDecrease
            entropyDecrease_temp(4) = current_entropy-left_entropy*left_size/label_size-right_entropy*right_size/label_size;
       
            
            [current_entropyDecrease, current_criteria] = max(entropyDecrease_temp);
            entropyDecreases(split_index,:) = [current_entropyDecrease,w1,w2,v1,v2,current_criteria];
       

    end

[~,best_entropyDecrease_index] = max(entropyDecreases(:,1));
 T(k,2:6) = entropyDecreases(best_entropyDecrease_index,2:6);

w1 = T(k,2);
w2 = T(k,3);
v1 = T(k,4);
v2 = T(k,5);

          
            
            
           
            I_w1w2 = uint8(zeros(size_figure(1),size_figure(2),Picturenumber_pertree));
        
         if w1>=0
           if w2>=0
             I_w1w2(w1+1:size_figure(1),1:size_figure(2)-w2,:) = I_current(1:size_figure(1)-w1,w2+1:size_figure(2),:);
             
           else
             I_w1w2(w1+1:size_figure(1),abs(w2)+1:size_figure(2),:) = I_current(1:size_figure(1)-w1,1:size_figure(2)-abs(w2),:);
             
           end
          elseif w1<0
           if w2>=0
             I_w1w2(1:size_figure(1)+w1,1:size_figure(2)-w2,:) = I_current(-w1+1:size_figure(1),w2+1:size_figure(2),:);
             
           else
             I_w1w2(1:size_figure(1)+w1,abs(w2)+1:size_figure(2),:) = I_current(-w1+1:size_figure(1),1:size_figure(2)-abs(w2),:);
           end
         
         end
         
         I_v1v2 = uint8(zeros(size_figure(1),size_figure(2),Picturenumber_pertree));
        
         if v1>=0
           if v2>=0
             I_v1v2(v1+1:size_figure(1),1:size_figure(2)-v2,:) = I_current(1:size_figure(1)-v1,v2+1:size_figure(2),:);
             
           else
             I_v1v2(v1+1:size_figure(1),abs(v2)+1:size_figure(2),:) = I_current(1:size_figure(1)-v1,1:size_figure(2)-abs(v2),:);
             
           end
          elseif v1<0
           if v2>=0
             I_v1v2(1:size_figure(1)+v1,1:size_figure(2)-v2,:) = I_current(-v1+1:size_figure(1),v2+1:size_figure(2),:);
             
           else
             I_v1v2(1:size_figure(1)+v1,abs(v2)+1:size_figure(2),:) = I_current(-v1+1:size_figure(1),1:size_figure(2)-abs(v2),:);
           end
         
         end

            
         if     entropyDecreases(best_entropyDecrease_index,6)==1;
            feature_label = uint8(I_w1w2.*I_v1v2);           
          
           
         elseif entropyDecreases(best_entropyDecrease_index,6)==2;        
            feature_label = uint8(I_w1w2.*(1-I_v1v2));            
            
       
         elseif entropyDecreases(best_entropyDecrease_index,6)==3;
            feature_label = uint8((1-I_w1w2).*I_v1v2);            
           
            
         else
            feature_label = uint8((1-I_w1w2).*(1-I_v1v2));            
         end

 left_label  = feature_label.*label; % label matrix for the pixels going to left
 right_label  = (1-feature_label).*label;  % label matrix for the pixels going to left  
             
      
      
T=buildtree_new_classification(T,left_label,left_child(k));

T=buildtree_new_classification(T,right_label ,right_child(k));
