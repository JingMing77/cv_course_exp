% 8点法求F
clc,
img = imread("521.jpg");
[h,w,c] = size(img);
if c==3
    img = rgb2gray(img);
end
imgl = img(1:h, 1:w/2);
imgr = img(1:h, w/2+1:w);
%%
% SIFT 提取并匹配特征点对
points1 = detectSIFTFeatures(imgl);
points2 = detectSIFTFeatures(imgr);
[features1,valid_points1] = extractFeatures(imgl,points1);
[features2,valid_points2] = extractFeatures(imgr,points2);
indexPairs = matchFeatures(features1,features2,"Unique",true);
matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);
% 把棋盘格周围匹配不好的点删去
n=1;
while(true)
    x = matchedPoints1.Location(n,1);
    y = matchedPoints1.Location(n,2);
    if x>50 && x<500 && y>50 && y<300
        matchedPoints1(n,:)=[];
        matchedPoints2(n,:)=[];
        n=n-1;
    end
    if(n>=matchedPoints1.Count)
        break;
    end
    n = n+1;
end
% 显示SIFT匹配结果
figure; 
showMatchedFeatures(imgl,imgr,matchedPoints1,matchedPoints2);

% 选取匹配点对计算
m1=[];
m2=[];
matchedPointsSize = size(matchedPoints1.Location);
for n=1:(matchedPointsSize(1)-1) % 删去坐标相同的点（对后续求解没有用）
    if ((matchedPoints1.Location(n,1)==matchedPoints1.Location(n+1,1))&&...
            (matchedPoints1.Location(n,2)==matchedPoints1.Location(n+1,2))) || ...
            ((matchedPoints2.Location(n,1)==matchedPoints2.Location(n+1,1))&&...
            (matchedPoints2.Location(n,2)==matchedPoints2.Location(n+1,2)))
        continue;
    end
    m1=[m1;matchedPoints1.Location(n,:)];
    m2=[m2;matchedPoints2.Location(n,:)];
end

% %%
% % 提取棋盘格角点 detectCheckerboardPoints
% [imagePoints,boardSize,pairsUsed] = detectCheckerboardPoints(imgl,imgr);
% m1 = imagePoints(:,:,1,1);
% m2 = imagePoints(:,:,1,2);

%%
m1 = [m1,ones(length(m1),1)];
m2 = [m2,ones(length(m2),1)];
val1 = m1(1:5,:);
val2 = m2(1:5,:);
m1(1:5,:)=[];
m2(1:5,:)=[];

% 构造线性方程组Ax=0
% ml'*F*mr=0
pointsNum = length(m1(:,1));
A = ones(pointsNum,9);
for n=1:pointsNum
    A(n,:)=[m2(n,1)*m1(n,:),m2(n,2)*m1(n,:),m1(n,:)];
end
% rank(A)=9 %因为噪声的存在

% 求解F矩阵
% 1 将系数矩阵 A 进行SVD分解
[~,~,V] = svd(A);
% 2 获得基本矩阵的估计值
F_hat = V(:,9);
F_hat = F_hat';
Fhat = [F_hat(1:3);F_hat(4:6);F_hat(7:9)];
[U,D,V] = svd(Fhat);
% 强制基本矩阵 rank(F) = 2
D(3,3) = 0;
F = U*D*V';

% 验证
pl=val1; % 匹配点对
pr=val2;
erl=zeros(1,5); % 误差
err=zeros(1,5); % 误差
for ii = 1:5 % 绘制内极线
    [imgr,erl(ii)] = drawEpipolarLines(pl(ii,1),pl(ii,2),pr(ii,1),pr(ii,2),F,imgr);
    [imgl,err(ii)] = drawEpipolarLines(pr(ii,1),pr(ii,2),pl(ii,1),pl(ii,2),F',imgl);
end
figure;
subplot(1,2,1), imshow(imgl); hold on; 
for ii = 1:5
    plot(pl(ii,1),pl(ii,2),'ro'); % 绘制匹配点对
end
subplot(1,2,2), imshow(imgr); hold on; 
for ii = 1:5
    plot(pr(ii,1),pr(ii,2),'ro');
end
erl_max = max(erl) % 最大误差
err_max = max(err)
erl_ave = sum(erl)/5 % 平均误差
err_ave = sum(err)/5
erl_var = sum((erl-erl_ave).^2)/5% 误差方差
err_var = sum((err-err_ave).^2)/5



