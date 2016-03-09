[FileName, PathName] = uigetfile('*.mat');
mat_path = fullfile(PathName, FileName);
s = load(mat_path);
prefix_path = uigetdir(PathName);
convert_mat_to_images(s.data, s.labels, prefix_path, 'images', 'images.csv');
