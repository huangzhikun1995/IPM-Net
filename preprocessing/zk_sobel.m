function [sobel] = sobel(src)
[Ay Ax dim] = size(src);
%ת��Ϊ�Ҷ�ͼ
if dim>1
    src = rgb2gray(src);
end
src = double(src);
sobel = zeros(Ay, Ax); 

for y = 2:(Ay-1)
    for x = 2:(Ax-1)
        sobel(y,x) = abs(src(y-1, x+1)-src(y-1, x-1) + 2*src(y, x+1) - 2*src(y, x-1) + src(y+1, x+1) - src(y+1, x-1)) + abs(src(y-1,x-1) - src(y+1,x-1) + 2*src(y-1,x)-2*src(y+1,x)+src(y-1,x+1)-src(y+1,x+1));                
        if sobel(y,x) < 60
            sobel(y,x) = 0;
        elseif sobel(y, x) > 200
            sobel(y, x) = 128;
        else
            sobel(y, x) = sobel(y,x);
        end
    end    
end
end%end of function

