# coding: utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt
import denoise
import imageio

def convert_to_gray():
    img_path = "./Lenna.png"
    img_gray = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    return img_gray

def saltpepper_noise(image, proportion):
    '''
    此函数用于给图片添加椒盐噪声
    image       : 原始图片
    proportion  : 噪声比例 
    '''
    image_copy = image.copy()
    # 求得其高宽
    img_Y, img_X = image.shape
    # 噪声点的 X 坐标
    X = np.random.randint(img_X,size=(int(proportion*img_X*img_Y),))
    # 噪声点的 Y 坐标
    Y = np.random.randint(img_Y,size=(int(proportion*img_X*img_Y),))
    # 噪声点的坐标赋值
    image_copy[Y, X] = np.random.choice([0, 255], size=(int(proportion*img_X*img_Y),))
    
    # 噪声容器
    sp_noise_plate = np.ones_like(image_copy) * 127
    # 将噪声给噪声容器
    sp_noise_plate[Y, X] = image_copy[Y, X]
    return image_copy, sp_noise_plate

def random_noise(image, proportion):
    '''
    此函数用于给图片添加随机噪声
    image       : 原始图片
    proportion  : 噪声比例 
    '''
    image_copy = image.copy()
    # 求得其高宽
    img_Y, img_X = image.shape
    # 噪声点的 X 坐标
    X = np.random.randint(img_X,size=(int(proportion*img_X*img_Y),))
    # 噪声点的 Y 坐标
    Y = np.random.randint(img_Y,size=(int(proportion*img_X*img_Y),))
    # 噪声点的坐标赋值
    image_copy[Y, X] = np.random.choice([i for i in range(30)]+[i for i in range(225,255)], size=(int(proportion*img_X*img_Y)))

    # 噪声容器
    sp_noise_plate = np.ones_like(image_copy) * 127
    # 将噪声给噪声容器
    sp_noise_plate[Y, X] = image_copy[Y, X]
    return image_copy, sp_noise_plate


if __name__ == "__main__":
    PSNR1_num=0
    PSNR2_num=0
    img=convert_to_gray()
    for i in range(1):
        a,b = random_noise(img, 0.1)
        imageio.imwrite("./pic1.png", a, format="bmp")

        test1=denoise.Picture("./pic1.png")
        test1.denoise(noise_type_list=[(denoise.PEPPER, denoise.SALT)], save_path="./denoise1_pic1.png")
        test2=denoise.Picture("./pic1.png", min_coef=20, max_coef=1.5)
        test2.denoise(noise_type_list=[denoise.RANDOM, (denoise.PEPPER, denoise.SALT)], save_path="./denoise2_pic1.png")

        d_a=imageio.imread("./denoise1_pic1.png")
        d_b=imageio.imread("./denoise2_pic1.png")
        PSNR1=denoise.cal_PSNR(img, d_a)
        PSNR2=denoise.cal_PSNR(img, d_b)
        print("PSNR1: {}".format(denoise.cal_PSNR(img, d_a)))
        print("PSNR2: {}".format(denoise.cal_PSNR(img, d_b)))
        print("1" if PSNR1>PSNR2 else "2")
        if PSNR1>PSNR2:
            PSNR1_num+=1
        else:
            PSNR2_num+=1
    print("PSNR1_num: {}\nPSNR2_num: {}".format(PSNR1_num, PSNR2_num))
    