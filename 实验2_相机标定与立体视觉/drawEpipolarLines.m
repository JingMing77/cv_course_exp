function [dst, er]=drawEpipolarLines(px,py,px2,py2,F,src)
% px,py:另一侧图像上的点
% px2,py2:src图像上的匹配点
% F:pl'Fpr=0    pr'F'pl=0
% 极线 au+bv+c=0 
% 误差 er = |ax+by+c|/sqrt(a^2+b^2)
p=[px,py,1];
ABC = p*F;
a = ABC(1);
b = ABC(2);
c = ABC(3);
er = abs(a*px2+b*py2+c)/sqrt(a^2 + b^2);
[m,n,ch]=size(src);
if ch==1
    dst = cat(3,src,src,src);
else
    dst = src;
end

for y = 1:n
    x = round((a*y)/(-b)+c/(-b));
    if x<1 || x>m
        break;
    end
    dst(x,y,1)=0;
    dst(x,y,2)=255;
    dst(x,y,3)=0;
end
for x = 1:m
    y = round((b*x)/(-a)+c/(-a));
    if y<1 || y>n
        break;
    end
    dst(x,y,1)=0;
    dst(x,y,2)=255;
    dst(x,y,3)=0;
end
end