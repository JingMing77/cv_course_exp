% Auto-generated by cameraCalibrator app on 27-Apr-2022
%-------------------------------------------------------


% Define images to process
imageFileNames = {'E:\计算机视觉\实验2\l\10.jpg',...
    'E:\计算机视觉\实验2\l\30.jpg',...
    'E:\计算机视觉\实验2\l\50.jpg',...
    'E:\计算机视觉\实验2\l\70.jpg',...
    'E:\计算机视觉\实验2\l\90.jpg',...
    'E:\计算机视觉\实验2\l\110.jpg',...
    'E:\计算机视觉\实验2\l\130.jpg',...
    'E:\计算机视觉\实验2\l\150.jpg',...
    'E:\计算机视觉\实验2\l\170.jpg',...
    'E:\计算机视觉\实验2\l\190.jpg',...
    'E:\计算机视觉\实验2\l\210.jpg',...
    'E:\计算机视觉\实验2\l\230.jpg',...
    'E:\计算机视觉\实验2\l\250.jpg',...
    'E:\计算机视觉\实验2\l\270.jpg',...
    'E:\计算机视觉\实验2\l\290.jpg',...
    'E:\计算机视觉\实验2\l\310.jpg',...
    'E:\计算机视觉\实验2\l\330.jpg',...
    'E:\计算机视觉\实验2\l\350.jpg',...
    'E:\计算机视觉\实验2\l\370.jpg',...
    'E:\计算机视觉\实验2\l\390.jpg',...
    'E:\计算机视觉\实验2\l\410.jpg',...
    'E:\计算机视觉\实验2\l\430.jpg',...
    'E:\计算机视觉\实验2\l\450.jpg',...
    'E:\计算机视觉\实验2\l\470.jpg',...
    'E:\计算机视觉\实验2\l\490.jpg',...
    'E:\计算机视觉\实验2\l\510.jpg',...
    'E:\计算机视觉\实验2\l\530.jpg',...
    'E:\计算机视觉\实验2\l\550.jpg',...
    'E:\计算机视觉\实验2\l\570.jpg',...
    'E:\计算机视觉\实验2\l\590.jpg',...
    };
% Detect calibration pattern in images
detector = vision.calibration.monocular.CheckerboardDetector();
[imagePoints, imagesUsed] = detectPatternPoints(detector, imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates for the planar pattern keypoints
squareSize = 10;  % in units of 'millimeters'
worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', true, 'EstimateTangentialDistortion', true, ...
    'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Visualize pattern locations
h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('MeasuringPlanarObjectsExample')
% showdemo('StructureFromMotionExample')
