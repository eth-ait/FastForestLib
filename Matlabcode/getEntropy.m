% This function computes the entropy of a label set. 


function [label_size,label_ent]=getEntropy(label,class_number)

% label: 3D label matrix
% label_ent: resulting entropy

p = zeros(1,class_number);
label_ent = 0;
label_size = sum(label(:)>=1);
if label_size==0
    label_ent=0;
    return;
end
for i = 1:class_number
p(i)= sum(label(:)==i)/label_size;
if p(i)>0
   label_ent = label_ent-p(i)*log(p(i));
end
end
