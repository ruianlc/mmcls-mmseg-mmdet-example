import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def cv_imread(file_path,color_patten=cv2.IMREAD_COLOR):
    """
    读取中文路径图片
    :ctime: 2022.06.20
    :param file_path:图片目录
    :param color_patten: 图片模式
    :return: 对应模式下的图片矩阵
    """
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),color_patten)
    return cv_img

def cv_imwrite(file_path,image, color_patten=cv2.COLOR_BGR2RGB, prefix='.bmp'):
    """
    写入中文路径图片
    :ctime: 2022.07.12
    :param file_path:图片目录
    :param color_patten: 图片模式
    :param prefix: 图片格式
    :return: 对应模式下的图片矩阵
    """
    if len(image.shape) == 3: # 彩色图，三维数据
        cv_img = cv2.imencode(prefix, cv2.cvtColor(image, color_patten))[1].tofile(file_path) # 保存带有中文路径的图片

    elif len(image.shape) == 2: # 灰度图，一维数据
        #cv_img_tmp = cv2.imencode(prefix, image)[1].tofile(file_path)
        cv_img = cv2.imencode(prefix, image)[1].tofile(file_path) # 保存带有中文路径的图片

    else:
        assert('wrong image dims!')
        return

    return cv_img

def cv_resize(img, scale_ratio):
    """
    修改图片尺寸
    :ctime: 2022.06.21
    :param img: 原始图片
    :param scale_ratio: 缩放比例
    :return: 缩放后的图片
    """
    width = int(img.shape[1] * scale_ratio)
    height = int(img.shape[0] * scale_ratio)

    img_dsize = cv2.resize(img, (width, height))

    return img_dsize

def computeMean(image, ignore_zeros=False):
    """
    计算灰度化图片像素平均值
    :param image: 灰度化图像
    :param ignore_zeros: 是否剔除值为0的像素点
    :return: 像素平均灰度值
    """
    if ignore_zeros:
        Y,X = np.nonzero(image)
        def select(im):
            return im[Y,X].ravel()
    else:
        select = np.ravel

    pixels = select(image)

    return np.mean(pixels)

def get_min(image, ignore_zeros=False):
    """
    计算灰度化图片像素平均值
    :param image: 灰度化图像
    :param ignore_zeros: 是否剔除值为0的像素点
    :return: 像素平均灰度值
    """
    if ignore_zeros:
        Y,X = np.nonzero(image)
        def select(im):
            return im[Y,X].ravel()
    else:
        select = np.ravel

    pixels = select(image)

    return np.min(pixels)

def cv_obtain_des(img):
    """
    剔除背景，提取目标本身
    :param img: 原始图像
    :return img_des: 剔除背景后的图像
    """
    # 高斯降噪
    img_blured = cv2.GaussianBlur(img, (7, 7), 0)
    #
    # 灰度图
    img_grey = cv2.cvtColor(img_blured, cv2.COLOR_RGB2GRAY)

    # 二值化图像
    dst, img_bin = cv2.threshold(img_grey, 29, 255, cv2.THRESH_BINARY)  # 自动计算分割阈值
    img_bin_3 = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

    # 提取目标本身
    img_fin = cv2.bitwise_and(img, img_bin_3) # 原始图像与目标掩码取交
    # img_des = cv2.cvtColor(img_fin, cv2.COLOR_BGR2RGB)

    return img_fin

def cv_kmeans(image_src, n_cluster=3):
    """
    对图像进行聚类
    :param image_src: 原始图像：彩色图或灰度图
    :param n_cluster: 类别数量
    :return: segmented_image - 聚类后的图像
             segmented_areas - 各个类别的面积
    """
    image = image_src.copy()

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    if len(image.shape) == 3: # 彩色图，三维数据
        pixel_values = image.reshape((-1, 3))
    elif len(image.shape) == 2: # 灰度图，一维数据
        pixel_values = image.reshape((-1, 1))
    else:
        assert('wrong image dims!')
        return

    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    # 聚成三类：背景、烟叶主色、杂色
    _, labels, (centers) = cv2.kmeans(pixel_values, n_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # 各类像素点个数（面积）
    segmented_list = np.unique(segmented_image).tolist()
    segmented_areas = [len((np.where(segmented_image == segmented_list[v]))[0]) for v in range(0, len(segmented_list))]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    # show the image
    #plt_imshow(segmented_image)

    return segmented_image, segmented_areas, centers.tolist()


def plt_imshow(image, figszie=(15,7)):
    plt.figure(figsize=figszie)
    if len(image.shape) == 3: # 彩色图，三维数据
        plt.imshow(image)
    elif len(image.shape) == 2: # 灰度图，一维数据
        plt.imshow(image,cmap='gray')
    else:
        assert('wrong image dims!')
        return

def cv_drawContours(img_src, img_bin, area_thresh):
    """
    寻找图像轮廓，并设置面积阈值标记轮廓
    :param img_src:
    :param img_bin:
    :param area_thresh:
    :return:
    """
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= area_thresh:  # 大轮廓，根据此阈值可以找到烟叶内部的孔洞
            cv_contours.append(contour)

    # 4、标出烟叶内部孔洞区域
    cv_contours.sort(key=lambda i: len(i), reverse=True)
    inner_contour = cv_contours#[1:]  # 剔除最大轮廓（仅标记内部轮廓）
    for contour in inner_contour:
        # x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(img_src, [contour], -1, (255, 25, 255), 4)

    plt.figure()
    plt.imshow(img_src)

    return inner_contour

def zh_ch(string):
    return string.encode('gbk').decode(errors='ignore')

def cv_hist(img, figszie=(15,7)):
    plt.figure(figsize=figszie)
    dims = img.ndim
    if dims == 3:
        ## 彩色图
        # calcHist
        # 参数1：要计算的原图，以方括号的传入，如：[img]。
        # 参数2：类似前面提到的dims，灰度图写[0]
        # 就行，彩色图R / G / B分别传入[0] / [1] / [2]。
        # 参数3：要计算的区域ROI，计算整幅图的话，写None。
        # 参数4：也叫bins, 子区段数目，如果我们统计0 - 255
        # 每个像素值，bins = 256；如果划分区间，比如0 - 15, 16 - 31…240 - 255
        # 这样16个区间，bins = 16。
        # 参数5：range, 要计算的像素值范围，一般为[0, 256)。
        color = ('r', 'g', 'b')
        hists = []
        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col, label=col)
            plt.xlim([0, 256])
            hists.append(hist)
        plt.legend(loc='best')
        #plt.show()

        return hists
    elif dims == 2:
        ## 灰度图
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.xlim([0, 256])
        #plt.show()

        return hist
    else:
        return

def get_peaks(srcArray,obtainNum=3):
    """
    画出数组直方图，寻找极值点
    :param srcArray: 原始一维数组
    :param obtainNum: 极值点数量
    :return: 找到极值点，并在直方图中标出
    """
    peakIdxs, _ = find_peaks(srcArray, threshold=0)
    peakVals = srcArray[peakIdxs]

    # 在极值点列表中的排序
    idxinpeakIdxs = peakVals.argsort()[::-1][0:obtainNum]

    # 在原始数组中的排序
    obtainIdxs = peakIdxs[idxinpeakIdxs]

    plt.plot(srcArray)
    plt.plot(obtainIdxs, srcArray[obtainIdxs], "x")
    # for i, j in zip(obtainIdxs, srcArray[obtainIdxs]):
    #     plt.text(i, j, '(%s,%s)' % (i, j), family='monospace', fontsize=12, color='r')
    plt.stem(obtainIdxs, srcArray[obtainIdxs])
    plt.xlim([0, 256])
    plt.show()