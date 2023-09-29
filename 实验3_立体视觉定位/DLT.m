clc, clear;
filename = 'data.xlsx'; % 读取表格信息
data_raw = xlsread(filename);
data_raw = data_raw(1:28, 1:8);
P_all = data_raw(:, 2:4);
img1_all = data_raw(:, 5:6);
img2_all = data_raw(:, 7:8);

% select more than 6 points
selNum = 6;
wrongsel = true;
while(wrongsel)
%     disp(wrongsel);
    p = randperm(28);
    select_row = p(1:selNum);
    select_P = [];
    select1 = [];
    select2 = [];
    for i = 1:selNum
        select_P = [select_P; P_all(select_row(i),:)];
        select1 = [select1; img1_all(select_row(i),:)];
        select2 = [select2; img2_all(select_row(i),:)];
    end
    n = cross(select_P(1,:)-select_P(2,:), select_P(1,:)-select_P(3,:));
    for ii = 4:selNum   % 确保6个点不能全部共面
        if norm(cross(n, select_P(ii,:))) > 1e-5
            wrongsel = 0;
            break;
        end
    end
end
[K1,R1,T1] = solveDLT(select_P, select1, selNum);
[K2,R2,T2] = solveDLT(select_P, select2, selNum);

M1 = K1*[R1,T1];
M2 = K2*[R2,T2];
remain_P = P_all;
remain1 = img1_all;
remain2 = img2_all;
remain_P(select_row,:) = [];
remain1(select_row,:) = [];
remain2(select_row,:) = [];

% 重投影误差
reproj_err_cal1 = zeros(selNum,1); % 相机1的 selNum个标定点的重投影误差
reproj_err_remain1 = zeros((28-selNum),1); % (28-selNum)个剩余点的重投影误差
reproj_err_cal2 = zeros(selNum,1); % 相机2的 selNum个标定点的重投影误差
reproj_err_remain2 = zeros((28-selNum),1); % (28-selNum)个剩余点的重投影误差
reproj_cal1 = zeros(selNum,2); % 相机1的 selNum个标定点
reproj_remain1 = zeros((28-selNum),2); % (28-selNum)个剩余点
reproj_cal2 = zeros(selNum,2); % 相机2的 selNum个标定点
reproj_remain2 = zeros((28-selNum),2); % (28-selNum)个剩余点
for i=1:selNum
    [reproj_cal1(i,:),reproj_err_cal1(i)] = reprojection_Error(M1, select_P(i,:), select1(i,:));
    [reproj_cal2(i,:),reproj_err_cal2(i)] = reprojection_Error(M2, select_P(i,:), select2(i,:));
end
for i=1:(28-selNum)
    [reproj_remain1(i,:),reproj_err_remain1(i)] = reprojection_Error(M1, remain_P(i,:), remain1(i,:));
    [reproj_remain2(i,:),reproj_err_remain2(i)] = reprojection_Error(M2, remain_P(i,:), remain2(i,:));
end

% 三维点重建
P3d_err_cal = zeros(selNum,1); % selNum个标定点的三维点重建误差
P3d_err_remain = zeros((28-selNum),1); % (28-selNum)个剩余点的三维点重建误差
P3d_cal = zeros(selNum,3); % selNum个标定点重建坐标
P3d_remain = zeros((28-selNum),3); % (28-selNum)个剩余点重建坐标
for i=1:selNum
    [P3d_cal(i,:),P3d_err_cal(i)] = cal3Dpoint(M1, M2, select1(i,:), select2(i,:), select_P(i,:));
end
for i=1:(28-selNum)
    [P3d_remain(i,:),P3d_err_remain(i)] = cal3Dpoint(M1, M2, remain1(i,:), remain2(i,:), remain_P(i,:));
end

% 结果写入Excel文件
result_P3d = zeros(28,3); 
result_reproj1 = zeros(28,2);
result_reproj2 = zeros(28,2);
result_P3d_err = zeros(28,1);
result_reproj1_err = zeros(28,1);
result_reproj2_err = zeros(28,1);
result_select = zeros(28,1);
remain_row = data_raw(:,1);
remain_row(select_row) = [];

result_P3d(select_row,:)=P3d_cal;
result_P3d(remain_row,:)=P3d_remain;
result_reproj1(select_row,:)=reproj_cal1;
result_reproj1(remain_row,:)=reproj_remain1;
result_reproj2(select_row,:)=reproj_cal2;
result_reproj2(remain_row,:)=reproj_remain2;
result_P3d_err(select_row,:)=P3d_err_cal;
result_P3d_err(remain_row,:)=P3d_err_remain;
result_reproj1_err(select_row,:)=reproj_err_cal1;
result_reproj1_err(remain_row,:)=reproj_err_remain1;
result_reproj2_err(select_row,:)=reproj_err_cal2;
result_reproj2_err(remain_row,:)=reproj_err_remain2;
result_select(select_row) = 1;

P3d_err_max = max(result_P3d_err);
P3d_err_min = min(result_P3d_err);
P3d_err_mean = mean(result_P3d_err);
P3d_err_var = var(result_P3d_err);
result_P3d_err = [result_P3d_err;P3d_err_max;P3d_err_min;P3d_err_mean;P3d_err_var];
reproj1_err_max = max(result_reproj1_err);
reproj1_err_min = min(result_reproj1_err);
reproj1_err_mean = mean(result_reproj1_err);
reproj1_err_var = var(result_reproj1_err);
result_reproj1_err = [result_reproj1_err;reproj1_err_max;reproj1_err_min;reproj1_err_mean;reproj1_err_var];
reproj2_err_max = max(result_reproj2_err);
reproj2_err_min = min(result_reproj2_err);
reproj2_err_mean = mean(result_reproj2_err);
reproj2_err_var = var(result_reproj2_err);
result_reproj2_err = [result_reproj2_err;reproj2_err_max;reproj2_err_min;reproj2_err_mean;reproj2_err_var];
P3d_err_cal_max = max(P3d_err_cal);
P3d_err_cal_min = min(P3d_err_cal);
P3d_err_cal_mean = mean(P3d_err_cal);
P3d_err_cal_var = var(P3d_err_cal);
P3d_err_remain_max = max(P3d_err_remain);
P3d_err_remain_min = min(P3d_err_remain);
P3d_err_remain_mean = mean(P3d_err_remain);
P3d_err_remain_var = var(P3d_err_remain);
result_P3d_err_cal = [P3d_err_cal_max;P3d_err_cal_min;P3d_err_cal_mean;P3d_err_cal_var];
result_P3d_err_remain = [P3d_err_remain_max;P3d_err_remain_min;P3d_err_remain_mean;P3d_err_remain_var];
reproj_err_cal1_max = max(reproj_err_cal1);
reproj_err_cal1_min = min(reproj_err_cal1);
reproj_err_cal1_mean = mean(reproj_err_cal1);
reproj_err_cal1_var = var(reproj_err_cal1);
reproj_err_cal2_max = max(reproj_err_cal2);
reproj_err_cal2_min = min(reproj_err_cal2);
reproj_err_cal2_mean = mean(reproj_err_cal2);
reproj_err_cal2_var = var(reproj_err_cal2);
reproj_err_remain1_max = max(reproj_err_remain1);
reproj_err_remain1_min = min(reproj_err_remain1);
reproj_err_remain1_mean = mean(reproj_err_remain1);
reproj_err_remain1_var = var(reproj_err_remain1);
reproj_err_remain2_max = max(reproj_err_remain2);
reproj_err_remain2_min = min(reproj_err_remain2);
reproj_err_remain2_mean = mean(reproj_err_remain2);
reproj_err_remain2_var = var(reproj_err_remain2);
result_reproj_err_cal1 = [reproj_err_cal1_max;reproj_err_cal1_min;reproj_err_cal1_mean;reproj_err_cal1_var];
result_reproj_err_cal2 = [reproj_err_cal2_max;reproj_err_cal2_min;reproj_err_cal2_mean;reproj_err_cal2_var];
result_reproj_err_remain1 = [reproj_err_remain1_max;reproj_err_remain1_min;reproj_err_remain1_mean;reproj_err_remain1_var];
result_reproj_err_remain2 = [reproj_err_remain2_max;reproj_err_remain2_min;reproj_err_remain2_mean;reproj_err_remain2_var];

xlswrite("data.xlsx", result_P3d, 'sheet1', 'I3');
xlswrite("data.xlsx", result_P3d_err, 'sheet1', 'L3');
xlswrite("data.xlsx", result_reproj1, 'sheet1', 'M3');
xlswrite("data.xlsx", result_reproj1_err, 'sheet1', 'O3');
xlswrite("data.xlsx", result_reproj2, 'sheet1', 'P3');
xlswrite("data.xlsx", result_reproj2_err, 'sheet1', 'R3');
xlswrite("data.xlsx", result_select, 'sheet1', 'S3');

xlswrite("data.xlsx", result_P3d_err_cal, 'sheet1', 'M31');
xlswrite("data.xlsx", result_P3d_err_remain, 'sheet1', 'N31');
xlswrite("data.xlsx", result_reproj_err_cal1, 'sheet1', 'P31');
xlswrite("data.xlsx", result_reproj_err_remain1, 'sheet1', 'Q31');
xlswrite("data.xlsx", result_reproj_err_cal2, 'sheet1', 'S31');
xlswrite("data.xlsx", result_reproj_err_remain2, 'sheet1', 'T31');

function [K,R,T] = solveDLT(P, uv, selNum)
% K,R,T: 相机的内参、外参矩阵
% P: selNum*3 矩阵，世界坐标系下点的坐标
% uv： selNum*2 矩阵，点对应的图像坐标
P1 = [P, ones(selNum,1)];
u = uv(:, 1);
v = uv(:, 2);
% 构造AL=0
A = [];
for i=1:selNum
    A1 = [P1(i,:), zeros(1, 4), -u(i).*P1(i,:)];
    A2 = [zeros(1, 4), P1(i,:), -v(i).*P1(i,:)];
    A = [A; A1; A2];
end
B = -A(:,12);
A(:, 12) = [];
L_hat = (A.'*A) \ A.'*B;
L_hat = [L_hat; 1].';
M = [L_hat(1:4); L_hat(5:8); L_hat(9:12)]; % M[3*4]
m1 = M(1, 1:3);
m2 = M(2, 1:3);
m3 = M(3, 1:3);
m34 = 1/norm(m3);
u0 = m34*m34*dot(m1,m3);
v0 = m34*m34*dot(m2,m3);
alphau = m34*m34*norm(cross(m1,m3));
alphav = m34*m34*norm(cross(m2,m3));
K = zeros(3,3);
K(1,1) = alphau;
K(1,3) = u0;
K(2,2) = alphav;
K(2,3) = v0;
K(3,3) = 1;
T = zeros(3,1);
T(1) = (m34.*(M(1,4)-u0))/alphau;
T(2) = (m34.*(M(2,4)-v0))/alphav;
T(3) = m34;
R = zeros(3,3);
R(1,:) = (m34.*(m1-u0.*m3))/alphau;
R(2,:) = (m34.*(m2-v0.*m3))/alphav;
R(3,:) = m34.*m3;
end

function [reproj, err] = reprojection_Error(M, P, uv)
% reproj: 重投影坐标[1*2]
% err: 重投影误差
% M： 3*4 投影矩阵 M=K[R,T]
% P: 1*3 向量，世界坐标系下点的坐标
% uv： 1*2 向量，点对应的图像坐标
reproj = M*[P';1];
reproj = reproj';
reproj = reproj/reproj(3);
reproj(3) = [];
err = norm(uv-reproj);
end

function [point3d, err] = cal3Dpoint(M1, M2, uv1, uv2, P)
% point3d: 重建三维点坐标[1*3]
% err: 三维点重建误差
% M1、M2： 相机1、2投影矩阵[3*4]
% uv1, uv2： 图像1、2坐标[1*2]
% P： 三维点真实坐标[1*3]
% 构造A*[X,Y,Z]'=B
A = zeros(4,3);
A(1,:) = [uv1(1)*M1(3,1)-M1(1,1), uv1(1)*M1(3,2)-M1(1,2), uv1(1)*M1(3,3)-M1(1,3)];
A(2,:) = [uv1(2)*M1(3,1)-M1(2,1), uv1(2)*M1(3,2)-M1(2,2), uv1(2)*M1(3,3)-M1(2,3)];
A(3,:) = [uv2(1)*M2(3,1)-M2(1,1), uv2(1)*M2(3,2)-M2(1,2), uv2(1)*M2(3,3)-M2(1,3)];
A(4,:) = [uv2(2)*M2(3,1)-M2(2,1), uv2(2)*M2(3,2)-M2(2,2), uv2(2)*M2(3,3)-M2(2,3)];
B = zeros(4,1);
B(1) = M1(1,4)-uv1(1)*M1(3,4);
B(2) = M1(2,4)-uv1(2)*M1(3,4);
B(3) = M2(1,4)-uv2(1)*M2(3,4);
B(4) = M2(2,4)-uv2(2)*M2(3,4);
point3d = (A.'*A)\A'*B; % [3*1]
point3d = point3d';
err = norm(point3d-P);
end