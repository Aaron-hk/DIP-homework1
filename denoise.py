# -*- encoding=utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
from sklearn.metrics import mean_squared_error


PEPPER = 0
SALT = 255
RANDOM = "random noise"
GAUSSIAN = "gaussian noise"


class Picture():
    def __init__(self, pic_path, min_coef=20, max_coef=1.5, filter_size=5):
        """read the picture

        Arguments:
            pic_path {str} -- [url of the picture]

        Keyword Arguments:
            min_coef {float} -- [compared with the neighborhood the threshold coef of the small gray value] (default: {1.8})
            max_coef {float} -- [compared with the neighborhood the threshold coef of the large gray value] (default: {1.2})
        """
        self.pic_name = pic_path[pic_path.rfind("\\")+1:]
        print(self.pic_name)
        self.image = imageio.imread(pic_path)
        self.m, self.n = self.image.shape
        self.min_coef = min_coef
        self.max_coef = max_coef
        self.sigma=0.54
        self.filter_size=filter_size

    def adaptive_median_filter(self, i, j, Smax, filter_size, noise_type=None):
        '''
        # noise_type=PEPPER means that only deal the PEPPER noise
        # noise_type=SALT means that only deal the SALT noise
        # noise_type=None or (PEPPER, SALT) means that deal the max and min noise in the filter area
        '''

        offset = (filter_size-1)/2
        f_x1 = int(max(0, j-offset))
        f_x2 = int(min(self.n-1, j+offset))
        f_y1 = int(max(0, i-offset))
        f_y2 = int(min(self.m-1, i+offset))
        filter_area = self.image[f_y1:f_y2+1, f_x1:f_x2+1]

        median = np.median(filter_area)
        z_min = np.min(filter_area)
        z_max = np.max(filter_area)

        if noise_type == PEPPER:
            z_max = 256
        elif noise_type == SALT:
            z_min = -1

        if median > z_min and median < z_max:
            if self.image[i][j] > z_min and self.image[i][j] < z_max:
                return self.image[i][j]
            else:
                return median
        else:
            filter_size += 2
            if filter_size <= Smax:
                return self.adaptive_median_filter(i, j, Smax, filter_size, noise_type)
            else:
                return median

    def adaptive_MF_random_nosie(self, i, j, Smax, filter_size, noise_type=None):
        """Consider the difference between the pixel value and the neighborhood value and
        judge the pixel is noise according to the difference 

        Arguments:
            Smax {int} -- [the maximum of the of the adaptive filter_size]

        Returns:
            [int] -- [the result of image[i][j]]
        """

        offset = (filter_size-1)/2
        f_x1 = int(max(0, j-offset))
        f_x2 = int(min(self.n-1, j+offset))
        f_y1 = int(max(0, i-offset))
        f_y2 = int(min(self.m-1, i+offset))
        filter_area = self.image[f_y1:f_y2+1, f_x1:f_x2+1]

        median = np.median(filter_area)
        z_min = np.min(filter_area)
        z_max = np.max(filter_area)
        if noise_type == PEPPER:
            z_max = 256
        elif noise_type == SALT:
            z_min = -1

        if median > z_min and median < z_max:
            if self.image[i][j] > z_min and self.image[i][j] < z_max and not self.is_noise(i, j, filter_area, noise_type):
                return self.image[i][j]
            else:
                return median
        else:
            filter_size += 2
            if filter_size <= Smax:
                return self.adaptive_MF_random_nosie(i, j, Smax, filter_size, noise_type)
            else:
                return median

    def is_noise(self, i, j, filter_area, noise_type):
        if noise_type == PEPPER:
            if self.image[i][j] < np.mean(filter_area)/self.min_coef:
                return True
            else:
                return False
        elif noise_type == SALT:
            if self.image[i][j] > np.mean(filter_area)*self.max_coef:
                return True
            else:
                return False
        elif noise_type == (PEPPER, SALT):
            if self.image[i][j] < np.mean(filter_area)/self.min_coef or self.image[i][j] > np.mean(filter_area)*self.max_coef:
                return True
            else:
                return False
        else:
            return False

    def gaussian_filter(self, i, j, filter_size):
        offset = int((filter_size-1)/2)
        f_x1 = int(max(0, j-offset))
        f_x2 = int(min(self.n-1, j+offset))
        f_y1 = int(max(0, i-offset))
        f_y2 = int(min(self.m-1, i+offset))
        filter_area = self.image[f_y1:f_y2+1, f_x1:f_x2+1]
        filter_coef = generate_gaussian_filter(filter_size, self.sigma)

        if filter_area.shape != (filter_size, filter_size):
            filter_coef = filter_coef[offset-(i-f_y1):offset+(
                f_y2-i)+1, offset-(j-f_x1):offset+(f_x2-j)+1]
            fill_value = np.sum(filter_area*filter_coef) + \
                (1-np.sum(filter_coef))*np.mean(filter_area)
        else:
            fill_value = np.sum(filter_area*filter_coef)
        return fill_value

    def denoise(self, noise_type_list, save_path):
        rst = np.zeros([self.m, self.n])

        if GAUSSIAN in noise_type_list:
            for i in range(self.m):
                for j in range(self.n):
                    rst[i][j] = self.gaussian_filter(i, j, filter_size=self.filter_size)
        elif RANDOM in noise_type_list:
            for i in range(self.m):
                for j in range(self.n):
                    rst[i][j] = self.adaptive_MF_random_nosie(
                        i, j, Smax=11, filter_size=self.filter_size, noise_type=noise_type_list[1])
        else:
            for i in range(self.m):
                for j in range(self.n):
                    rst[i][j] = self.adaptive_median_filter(
                        i, j, Smax=11, filter_size=self.filter_size, noise_type=noise_type_list[0])
        rst=rst.astype(np.uint8)
        imageio.imwrite(save_path, rst, format='bmp')

        a=imageio.imread(save_path)
        for i in range(self.m):
            for j in range(self.n):
                if a[i][j]!=rst[i][j]:
                    print("="*15)

def show_histogram(image, save_path=None):
    topn = 20
    # plt.hist(image, bins=256, density=1)
    # plt.show()
    # return
    m, n = image.shape
    histogram = [0 for i in range(256)]
    for i in range(m):
        for j in range(n):
            histogram[image[i][j]] += 1

    # # normalize
    histogram = [i/(m*n) for i in histogram]

    histuple_list = []
    for i in range(256):
        histuple_list.append((i, histogram[i]))
    histuple_list = sorted(histuple_list, key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(30,10))
    plt.grid(True)
    plt.bar([str(i[0]) for i in histuple_list[0:topn]], [i[1]
                                                         for i in histuple_list[0:topn]])
    plt.bar([str(i) for i in range(0,10)], [histogram[i] for i in range(0,10)])
    plt.bar([str(i) for i in range(246,256)], [histogram[i] for i in range(246,256)])

    if save_path != None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def generate_gaussian_filter(filter_size, sigma):
    centor = (filter_size-1)/2
    filter_coef = np.zeros([filter_size, filter_size])
    coef1 = 1/(2*math.pi*sigma*sigma)
    for i in range(filter_size):
        for j in range(filter_size):
            exp_num = math.pow(i-centor, 2)+math.pow(j-centor, 2)
            exp_num = -(exp_num/2/sigma/sigma)
            filter_coef[i][j] = coef1*math.exp(exp_num)
    return filter_coef/np.sum(filter_coef)


def cal_PSNR(src_image, image):
    MAX = math.pow(2, 8)-1
    MSE = mean_squared_error(src_image, image)
    m, n = src_image.shape
    MSE = max(MSE, 1/m*n)
    PSNR = 10*math.log(math.pow(MAX, 2)/MSE, 10)
    return PSNR


# 23: 111-115
if __name__ == "__main__":
    noise_dir = "./noise_picture"
    if not os.path.exists(noise_dir):
        os.mkdir(noise_dir)
    denoise_dir = "./denoise_picture"
    if not os.path.exists(denoise_dir):
        os.mkdir(denoise_dir)
    histogram_dir = "./histogram"
    if not os.path.exists(histogram_dir):
        os.mkdir(histogram_dir)
    start=111
    end=116

    # show the histogram of noise picture
    for i in range(start, end):
        pic = str(i)+".jpeg"
        pic_path = os.path.join(noise_dir, pic)
        his_save_path = os.path.join(histogram_dir, str(i)+"_hist"+".jpeg")
        a = imageio.imread(pic_path)
        show_histogram(a, his_save_path)

    # test some operations
    pic_noise_dict = {}
    pic_noise_dict[111] = [PEPPER]
    pic_noise_dict[112] = [PEPPER]
    pic_noise_dict[113] = [GAUSSIAN]
    pic_noise_dict[114] = [(PEPPER, SALT)]
    pic_noise_dict[115] = [SALT]
    for i in range(start, end):
        pic = str(i)+".jpeg"
        pic_path = os.path.join(noise_dir, pic)
        save_path = os.path.join(denoise_dir, "test_"+pic)
        test = Picture(pic_path)
        test.denoise(noise_type_list=pic_noise_dict[i], save_path=save_path)
    for i in range(start, end):
        pic = "test_"+str(i)+".jpeg"
        pic_path = os.path.join(denoise_dir, pic)
        his_save_path = os.path.join(
            histogram_dir, "test_denoise_"+str(i)+"_hist"+".jpeg")
        a = imageio.imread(pic_path)
        show_histogram(a, his_save_path)

    ## final
    pic_noise_dict = {}
    pic_noise_dict[111] = [RANDOM, PEPPER]
    pic_noise_dict[112] = [RANDOM, PEPPER]
    pic_noise_dict[113] = [GAUSSIAN]
    pic_noise_dict[114] = [RANDOM, (PEPPER, SALT)]
    pic_noise_dict[115] = [RANDOM, SALT]
    pic_coef={}
    pic_coef[111]=[1.8, None]
    pic_coef[112]=[4.5, None]
    pic_coef[113]=[None, None]
    pic_coef[114]=[4.25, 1.2]
    pic_coef[115]=[None, 1.1]
    for i in range(start, end-1):
        pic = str(i)+".jpeg"
        pic_path = os.path.join(noise_dir, pic)
        save_path = os.path.join(denoise_dir, pic)
        test = Picture(pic_path, min_coef=pic_coef[i][0], max_coef=pic_coef[i][1])
        test.denoise(noise_type_list=pic_noise_dict[i], save_path=save_path)

    i=115
    pic = str(i)+".jpeg"
    pic_path = os.path.join(noise_dir, pic)
    save_path = os.path.join(denoise_dir, pic)
    test = Picture(pic_path, min_coef=pic_coef[i][0], max_coef=pic_coef[i][1], filter_size=3)
    test.denoise(noise_type_list=pic_noise_dict[i], save_path=save_path)

    for i in range(start, end):
        pic = str(i)+".jpeg"
        pic_path = os.path.join(denoise_dir, pic)
        his_save_path = os.path.join(
            histogram_dir, "denoise_"+str(i)+"_hist"+".jpeg")
        a = imageio.imread(pic_path)
        show_histogram(a, his_save_path)