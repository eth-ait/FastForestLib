function p=decideTree_classification(I,x_u1,x_u2,T_tree)


% T_tree: decision tree
% I: current binary image
% p: result leaf node probability distribution of both parts and gestures for current
% candidate pixel


k=1; % node index
[d ,~] = index2depth(k); % depth and column
global size_figure; % size of the image
global Depth_final; % final depth
global Gestures_number;
% p = zeros(1,1,Gestures_number); % p: initialization of result leaf node probability distribution of both parts and gestures for current


%% pass the pixel into the decition tree


while d<Depth_final
    

%    if T_tree(k,1)==1
%    for gesture_index = 1:Gestures_number
%       p = T_tree(k,2);
%    end
%    
%    return
  
%    else
%  if T_tree(k,10) == 0 || isnan(T_tree(right_child(k),2))
 if T_tree(k,6) == 0 
     k=left_child(k);
     [d, ~]=index2depth(k);
 else
     
    w1 = -T_tree(k,2);
    w2 = T_tree(k,3);
    v1 = -T_tree(k,4);
    v2 = T_tree(k,5);      
           
    if (x_u1 + w1)<1 || (x_u1 + w1)> size_figure(1) || (x_u2 + w2)<1 || (x_u2 + w2)> size_figure(2)
       feature_w = 0;
    else
       feature_w = I(x_u1 + w1,x_u2 + w2);
    end
           
    if (x_u1 + v1)<1 || (x_u1 + v1)> size_figure(1) || (x_u2 + v2)<1 || (x_u2 + v2)> size_figure(2)
       feature_v = 0;
    else
       feature_v = I(x_u1 + v1,x_u2 + v2);
    end
      
    
    
    if T_tree(k,6)==1;
       feature_label = feature_w*feature_v;           
       if feature_label==1; %  pixel going to left
          k=left_child(k);
          [d, ~]=index2depth(k);
       else
          k=right_child(k);
          [d, ~]=index2depth(k);  %  pixel going to right
       end
     elseif T_tree(k,6)==2;        
        feature_label = feature_w*(1-feature_v);            
        if feature_label==1; %  pixel going to left
           k=left_child(k);
           [d, ~]=index2depth(k);
        else
           k=right_child(k);
           [d, ~]=index2depth(k);  %  pixel going to right
        end      
     elseif T_tree(k,6)==3;
         feature_label = (1-feature_w)*feature_v;            
         if feature_label==1; %  pixel going to left
            k=left_child(k);
            [d, ~]=index2depth(k);
         else
            k=right_child(k);
            [d, ~]=index2depth(k);  %  pixel going to right
         end
     elseif T_tree(k,6)==4;
         feature_label = (1-feature_w)*(1-feature_v);            
         if feature_label==1; %  pixel going to left
            k=left_child(k);
            [d, ~]=index2depth(k);
         else
            k=right_child(k);
            [d, ~]=index2depth(k);  %  pixel going to right
         end         
    end
    
  end
    
end

 



%%
if d==Depth_final
   
   
   p = T_tree(k,7:6+ Gestures_number);
%       p1 = T_tree(k,2);
% %       [~,a] = max(T_tree(k,11:13));
%       
%      [~,tem] = max(T_tree(k,11:13));
%       p2 = T_tree(k,tem+2);
% 
%      [~,tem] = max(T_tree(k,11:15));
%       p2 = T_tree(k,tem+15);
%       temp = (T_tree(k,11)+T_tree(k,12)+T_tree(k,13));
%       p2 = T_tree(k,3)*T_tree(k,11)/temp + T_tree(k,4)*T_tree(k,12)/temp + T_tree(k,5)*T_tree(k,13)/temp;
%        p2 = p1;
   
   return
end
end
