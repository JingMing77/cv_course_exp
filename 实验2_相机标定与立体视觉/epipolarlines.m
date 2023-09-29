clc,
% 计算F矩阵
R = stereoParams.RotationOfCamera2; % 相机2相对于相机1的偏移矩阵
T = stereoParams.TranslationOfCamera2; % 相机2相对于相机1的旋转矩阵，需要转置之后才能使用
Kl = (stereoParams.CameraParameters1.IntrinsicMatrix).'; % 内参矩阵
Kr = (stereoParams.CameraParameters2.IntrinsicMatrix).';
% F=(calF(R,T,Kl,Kr))';
F=(stereoParams.FundamentalMatrix)';
% 选取 5 组匹配点对
img = imread("521.jpg");
[h,w,c] = size(img);
imgl = img(1:h, 1:w/2,:);
imgr = img(1:h, w/2+1:w,:);
[imagePoints,boardSize,pairsUsed] = detectCheckerboardPoints(imgl,imgr);
pl=ones(5,3); % 匹配点对
pr=ones(5,3);
erl=zeros(1,5); % 误差
err=zeros(1,5); % 误差
for ii = 1:5
    pl(ii,1)=imagePoints(ii*7,1,1,1);
    pl(ii,2)=imagePoints(ii*7,2,1,1);
    pr(ii,1)=imagePoints(ii*7,1,1,2);
    pr(ii,2)=imagePoints(ii*7,2,1,2);
    [imgr,erl(ii)] = drawEpipolarLines(pl(ii,1),pl(ii,2),pr(ii,1),pr(ii,2),F,imgr); % 绘制内极线
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