N = 1000;
W = 100;
H = 100;

numClasses = 4;
maxValue = 255;

data = zeros(W, H, N);
labels = -ones(W, H, N);

for i = 1:numClasses
    indices = (1 + (i-1) * (N / numClasses)) : (i * (N / numClasses));
    value = i / numClasses * maxValue;
    data(1:W/2, 1:H/2, indices) = value;
    labels(1:W/2, 1:H/2, indices) = i - 1;
end

save('syntheticData.mat', 'data', 'labels');
