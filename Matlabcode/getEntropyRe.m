function [label_size,label_ent] = getEntropyRe(depth_map) 

depth_map(depth_map == 0) = [];
depth_map = single(depth_map);
label_ent = log(std(depth_map));
depth_map = logical(depth_map);
label_size = sum(depth_map);
end

