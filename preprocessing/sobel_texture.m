clear all;
clear
clc;

dataroot = '.';

imglist = dir(fullfile(dataroot, 'testA', '*.png'));
masklist = dir(fullfile(dataroot, 'testA_feature', '*.png'));
savepath = [dataroot, '/testA_texture'];
len = length(imglist);
if len ~= length(masklist)
    error('The number of files in the two folders is not equal!');
end
if ~exist(savepath)
    mkdir(savepath);
end

for i = 1: len
    file_name = imglist(i).name;
    img = imread([dataroot, '/testA/', file_name]);
    mask = imread([dataroot, '/testA_feature/', file_name]);
    
    % resize source images
    [m, n] = size(mask);
    img = imresize(img, [m, n]);
    feature = uint8(zeros(m, n, 3));
    for d = 1 : 3
        feature(:, :, d) = img(:, :, d) .* ((255 - mask)/255);
    end

    % Use the sobel operator to find the gradient
    sobel = uint8(zk_sobel(feature));
    savename = [savepath, '/', file_name]
    imwrite((sobel./2), savename);
end


    

