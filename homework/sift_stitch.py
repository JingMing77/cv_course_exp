# coding=utf-8
import os
import cv2
import numpy as np


# 读取图像（存储顺序是从左到右）
def read_images(directory_name, w=None, h=None):
    imgs = []
    for filename in os.listdir(directory_name):
        # img is used to store the image data
        if filename[-3:] == 'jpg':
            img = cv2.imread(directory_name + "/" + filename)
            imgs.append(img)

    # 如果输入图像尺寸不一致需要矫正图像尺寸
    if len(imgs) >= 2 and w is not None:
        for i in range(len(imgs)):
            imgs[i] = cv2.resize(imgs[i], dsize=(w, h))

    return imgs


# SIFT 特征点检测
def detectAndDescribe(image):
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])

    return kps, features


# 特征点匹配
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # 存储两个点在featuresA, featuresB中的索引值
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # 当筛选后的匹配对大于4时，计算单应矩阵H
    if len(matches) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # 计算视角变换矩阵
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        # 返回结果
        return matches, H, status

    # 如果匹配对小于4时，返回None
    return None


# 显示匹配点对
def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # 联合遍历，画出匹配对
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # 当点对匹配成功时，画到可视化图上
        if s == 1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 255), 1)

    cv2.namedWindow('matches', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('matches', vis)
    cv2.waitKey(0)

    return None


def warp(src, H, height=None, width=None):
    h0 = src.shape[0]
    w0 = src.shape[1]
    h = height
    w = width
    if h is None or w is None:
        W = np.mat(H)
        XY1 = np.array([[w0-1], [0], [1]])
        XY1 = np.mat(XY1)
        x1, y1, z1 = W.dot(XY1)
        x1 = np.round(x1[0][0] / z1[0][0])
        y1 = np.round(y1 / z1)
        XY2 = np.array([[w0 - 1], [h0 - 1], [1]])
        XY2 = np.mat(XY2)
        x2, y2, z2 = W.dot(XY2)
        x2 = np.round(x2[0][0] / z2[0][0])
        y2 = np.round(y2 / z2)
        w = int((max(x1, x2)))
        h = int(1.2 * abs(y2 - y1))

    result = np.zeros((h, w, 3), dtype=np.uint8)
    # opencv上面的部分会被裁掉，上半截需要自己算
    W = np.mat(H)
    W = W.I  # 返回原图中找对应的点，所以要先求逆
    delta = (h - h0) // 2  # 变换后向下平移delta，以显示全部变换后的图像
    for i in range(delta):
        for j in range(w):
            XY1 = np.array([[j], [i - delta], [1]])
            # j,i 的位置在opencv中被转置了
            XY1 = np.mat(XY1)
            x, y, z = W.dot(XY1)
            x = np.round(x[0][0] / z[0][0])
            y = np.round(y / z)
            # 检查是否在原图范围内
            if x < 0 or x >= w0 or y < 0 or y >= h0:
                continue

            result[i, j, 0] = src[int(y), int(x), 0]
            result[i, j, 1] = src[int(y), int(x), 1]
            result[i, j, 2] = src[int(y), int(x), 2]
    # 剩下的用opencv的，否则计算时间太长
    ttt = cv2.warpPerspective(src, H, dsize=(w, h - delta))
    result[delta:h, :, :] = ttt

    return result, delta


# 计算校正后的长方形的图像
def correct(src, h, delta=0, flag='perspect', imgNum=1):
    h0 = src.shape[0]
    w0 = src.shape[1]
    # 计算图像的宽度
    w = w0 - 1
    for i in range(w0):
        if np.any(src[:, w0 - 1 - i, :]):
            w = w0 - i - 1
            break
    result = np.zeros((h, w, 3), dtype=np.uint8)
    # 如果没有给定delta（向下偏移量），则需要自行计算
    if delta == 0:
        for i in range(h0):
            if np.any(src[i, 0, :]):
                delta = i
                break

    # 把每一列resize，效果很差，后面的会皱成一团
    if flag == 'resize':
        for i in range(w):
            s = delta
            t = h + delta
            for j in range(delta):
                if not np.any(src[delta - j, i, :]):
                    s = delta - j
                    break
            for j in range(delta):
                if not np.any(src[h + delta + j, i, :]):
                    t = h + delta + j
                    break
            vec = src[s:t, i: i + 1, :]
            result[:, i: i + 1, :] = cv2.resize(vec, dsize=(1, h))

    # 把有图像的部分，（大概是一个类似梯形的东西）
    # 通过透视投影变换成长方形
    if flag == 'perspect':
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        # 右上角
        for i in range(h0):
            if np.any(src[i, :, :]):
                y1 = i
                for j in range(w0):
                    if np.any(src[i, j, :]):
                        x1 = j
                        break
                break
        # 右下角
        for i in range(h0):
            if np.any(src[h0 - 1 - i, :, :]):
                y2 = h0 - 1 - i
                for j in range(w0):
                    if np.any(src[y2, j, :]):
                        x2 = j
                        break
                break
        # 原图坐标
        pts1 = np.float32([[0, delta], [0, delta+h], [x1, y1], [x2, y2]])
        # 获得目标图坐标（与原图坐标对应起来）
        pts2 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(src, M, (w, h))
        # cv2.imshow("r22", result)
        # cv2.waitKey(0)

        # 仅仅做透视投影，中间会出现黑边，
        # 为了消除黑边，可以多找几个点进行插值，
        # n张图像则需要在中间找n-1个点
        if imgNum > 1:
            result2 = np.zeros((h, w, 3), dtype=np.uint8)
            dx = w//imgNum
            tx = 0
            ty1 = 0
            ty2 = h
            for i in range(imgNum):
                pts1 = np.float32([[0, ty1], [0, ty2]])
                pts2 = np.float32([[0, 0], [0, h]])
                tx = tx + dx
                for j1 in range(h):
                    if np.any(result[j1, tx-1, :]):  # 不是黑边
                        ty1 = j1 - 1
                        break
                for j2 in range(h):
                    if np.any(result[h - 1 - j2, tx-1, :]):
                        ty2 = h - j2
                        break

                pts1 = np.vstack((pts1, np.float32([[dx, ty1], [dx, ty2]])))
                pts2 = np.vstack((pts2, np.float32([[dx, 0], [dx, h]])))

                M = cv2.getPerspectiveTransform(pts1, pts2)
                result2[:, tx-dx:tx, :] = cv2.warpPerspective(result[:, tx-dx:tx, :], M, (dx, h))
                cv2.imshow("r2", result2)
                cv2.waitKey(0)
            result = result2
    return result


def stitch_1(images, imgNum, ratio=0.75, reprojThresh=4.0, showMatches=False, chazhi=True):
    assert imgNum >= 2

    result = images[imgNum - 1]
    for i in range(imgNum - 1):
        # 获取输入图片
        imageA = result
        imageB = images[imgNum - 2 - i]
        # 检测A、B图片的SIFT关键特征点，并计算特征描述子
        (kpsA, featuresA) = detectAndDescribe(imageA)
        (kpsB, featuresB) = detectAndDescribe(imageB)
        # 匹配两张图片的所有特征点
        M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        if M is None:
            return None
        (matches, H, status) = M

        # 显示匹配结果，并连线表示
        if showMatches:
            drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

        # 将当前图像与上一张图像进行拼接
        result = cv2.warpPerspective(
            imageA, H,
            (int(1.05 * (imageA.shape[1] + imageB.shape[1])), int(1.1 * imageA.shape[0])))

        cv2.namedWindow('result', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('result', result)
        cv2.waitKey(0)
        if not chazhi:
            result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        if chazhi:
            # 重叠部分(x1,x2) 插值
            x1 = 0
            x2 = imageB.shape[1]
            for xx in range(imageA.shape[1] + imageB.shape[1]):
                # x1 = x1 + 1
                if np.any(result[imageA.shape[0] // 2, xx + 3, :]):
                    x1 = xx
                    break

            for xx in range(x2 - x1):
                rB = (x2 - x1 - xx) / (x2 - x1)
                rA = 1. - rB
                temp = rA * result[:imageB.shape[0], (xx + x1), :] + rB * imageB[:imageB.shape[0], (xx + x1), :]
                result[:imageB.shape[0], xx + x1, :] = np.array(temp).astype(np.uint8)

            result[0:imageB.shape[0], 0:x1] = imageB[0:imageB.shape[0], 0:x1]

        cv2.namedWindow('result', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('result', result)
        cv2.waitKey(0)

    cv2.namedWindow('result', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    # 返回匹配结果
    return result


def stitch_2(images, imgNum, ratio=0.75, reprojThresh=4.0, showMatches=False):
    assert imgNum >= 2

    result = images[imgNum - 1]
    for i in range(imgNum - 1):
        # 获取输入图片
        imageA = result
        imageB = images[imgNum - 2 - i]
        # 检测A、B图片的SIFT关键特征点，并计算特征描述子
        (kpsA, featuresA) = detectAndDescribe(imageA)
        (kpsB, featuresB) = detectAndDescribe(imageB)
        # 匹配两张图片的所有特征点
        M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        if M is None:
            return None
        (matches, H, status) = M

        # 显示匹配结果，并连线表示
        if showMatches:
            drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

        # 将当前图像与上一张图像进行拼接
        # 1. 计算透视变换后的图像
        result, delta = warp(imageA, H)

        cv2.namedWindow('result', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('result', result)
        cv2.waitKey(0)

        # 2. 计算重叠部分(x1,x2)
        x1 = 0
        x2 = imageB.shape[1]
        for xx in range(imageA.shape[1] + imageB.shape[1]):
            # x1 = x1 + 1
            if np.any(result[imageA.shape[0] // 2, xx, :]):
                x1 = xx
                break
        # 3. 给左边的图像加黑框，往下平移
        imageB2 = np.zeros((result.shape[0], imageB.shape[1], 3), dtype=np.uint8)
        imageB2[delta:delta + imageB.shape[0], :, :] = imageB
        # 4. 对重叠部分(x1,x2) 进行插值
        for xx in range(x2 - x1):
            rB = (x2 - x1 - xx) / (x2 - x1)
            rA = 1. - rB
            temp = rA * result[:imageB2.shape[0], (xx + x1), :] + rB * imageB2[:imageB2.shape[0], (xx + x1), :]
            result[:imageB2.shape[0], xx + x1, :] = np.array(temp).astype(np.uint8)

        result[0:imageB2.shape[0], 0:x1] = imageB2[0:imageB2.shape[0], 0:x1]

        cv2.namedWindow('result', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('result', result)
        cv2.waitKey(0)

        # 对变换后的图像(x2, w)部分进行校正
        # 最后希望的shape:(images[0].shape[0], width, 3)
        # width = 把黑边裁掉，有图像的部分的宽度
        result = correct(result, images[0].shape[0], imgNum=2)

    cv2.imwrite("result/result02.jpg", result)
    cv2.namedWindow('result', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    # 返回匹配结果
    return result


def main():
    images = read_images("E:/computer_vision/homework", w=256, h=360)
    # 1. 直接拼接+插值 会出现最边缘的图片越来越模糊的现象
    # stitch_1(images, len(images), ratio=0.9, reprojThresh=3.0, showMatches=False, chazhi=True)
    # 2. 改进版，输出的图像是长方形的
    stitch_2(images, len(images), ratio=0.9, reprojThresh=3.0, showMatches=True)

    # ss = cv2.imread("result/result01.jpg")
    # rr = correct(ss, 359, imgNum=4)
    # cv2.imwrite("result/终极result.jpg", rr)
    # cv2.imshow("rr", rr)
    # cv2.waitKey(0)


if __name__ == "__main__":
    main()
