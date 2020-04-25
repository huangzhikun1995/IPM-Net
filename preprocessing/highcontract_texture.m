clear all;
clear
clc;

dataroot = '.';

imglist = dir(fullfile(dataroot, 'trainA', '*.png'));
masklist = dir(fullfile(dataroot, 'trainA_feature', '*.png'));
savepath = [dataroot, '/trainA_highcontract'];
len = length(imglist);
if len ~= length(masklist)
    error('The number of files in the two folders is not equal!');
end
if ~exist(savepath)
    mkdir(savepath);
end

for i = 1: len
    file_name = imglist(i).name;
    img = imread([dataroot, '/trainA/', file_name]);
    mask = imread([dataroot, '/trainA_feature/', file_name]);
    
    % resize source images
    [m, n] = size(mask);
    img = imresize(img, [m, n]);
    contract = uint8(zeros(m, n));
    bluelayer = img(:,:,3) .* (1-mask/255);
    w=fspecial('gaussian',[10 10],2);
    contract = bluelayer - imfilter(bluelayer,w);
%     for x = 1: m
%         for y = 1: n
%             if mask(x, y) == 0 
%                 contract(x, y) = contract(x, y) + 127;
%             end
%         end
%     end   

    % Use the sobel operator to find the gradient
%     figure(1);
%     subplot(121);imshow(contract);
%     subplot(122);imshow(mask);
    savename = [savepath, '/', file_name]
    imwrite(contract, savename);
end


    

