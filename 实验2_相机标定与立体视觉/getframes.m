fileName = '标定与基本矩阵计算.avi'; 
obj = VideoReader(fileName);
numFrames = obj.NumFrames;% 帧的总数
 for k = 10:20:numFrames % 读取数据
     frame = read(obj,k);
     [h,w,c] = size(frame);
     frame_l = frame(1:h, 1:w/2,:);
     frame_r = frame(1:h, w/2+1:w,:);
     imwrite(frame_r,strcat(num2str(k),'.jpg'),'jpg');% 保存帧
 end
 