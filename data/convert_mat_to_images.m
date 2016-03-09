function convert_mat_to_images(data, labels, prefix, image_path, image_list_csv_filename)

if nargin < 2
    prefix = 'training_data/';
    image_list_csv_filename = 'training_images.csv';
end
full_image_list_csv_filename = fullfile(prefix, image_list_csv_filename);

training_images_csv_file = fopen(full_image_list_csv_filename, 'w');
label_counts = zeros(1, 3);

mkdir(fullfile(prefix, image_path));

numClasses = max(labels(:)) + 1;
my_labels = labels;
my_labels(labels < 0) = numClasses;

for i = 1:size(data, 3)
    data_image_filename = fullfile(image_path, ['data_image_', int2str(i), '.png']);
    label_image_filename = fullfile(image_path, ['label_image_', int2str(i), '.png']);
    data_image = uint16(squeeze(data(:, :, i)));
%     data_image = data_image / max(data_image(:));
    label_image = uint16(squeeze(my_labels(:, :, i)));
%     label_image = label_image / max(label_image(:));

    for j = 0:max(label_image(:))
        if numel(label_counts) < j+1
            label_counts(j+1) = 0;
        end
        label_counts(j+1) = label_counts(j+1) + sum(label_image(:) == j);
    end

    imwrite(data_image, fullfile(prefix, data_image_filename));
    imwrite(label_image, fullfile(prefix, label_image_filename));
    fprintf(training_images_csv_file, '%s, %s\n', data_image_filename, label_image_filename);
end
fclose(training_images_csv_file);

end