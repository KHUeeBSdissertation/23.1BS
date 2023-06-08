import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import os.path
from tqdm import tqdm
import math
import cv2


def load_images(img_dir):
    imgs = []
    img_list = cv2.imread(img_dir)
    imgs.append(img_list)
    # for file in tqdm(os.listdir(img_dir)):
    #     imgs.append(np.array(Image.open(img_dir + file)))
    # both input images are from 0-->255
    return imgs

def generate_histogram(img, do_print):
    """
    @params: img: can be a grayscale or color image. We calculate the Normalized histogram of this image.
    @params: do_print: if or not print the result histogram
    @return: will return both histogram and the grayscale image 
    """
    # print(img)
    # if img.shape[2] == 3: # img is colorful, so we convert it to grayscale
    #     gr_img = np.mean(img, axis=-1)
    # else:
    #     gr_img = img
    gr_img = img
    '''now we calc grayscale histogram'''
    gr_hist = np.zeros([256])

    for x_pixel in range(gr_img.shape[0]):
        for y_pixel in range(gr_img.shape[1]):
            pixel_value = int(gr_img[x_pixel, y_pixel])
            gr_hist[pixel_value] += 1
            
    '''normalizing the Histogram'''
    gr_hist /= (gr_img.shape[0] * gr_img.shape[1])
    if do_print:
        print_histogram(gr_hist, name="n_h_img", title="Normalized Histogram")
    return gr_hist, gr_img

def print_histogram(_histrogram, name, title):
    plt.figure()
    plt.title(title)
    plt.plot(_histrogram, color='#ef476f')
    plt.bar(np.arange(len(_histrogram)), _histrogram, color='#b7b7a4')
    plt.ylabel('Number of Pixels')
    plt.xlabel('Pixel Value')
    plt.savefig("hist_" + name)

def equalize_histogram(img, histo, L):
    eq_histo = np.zeros_like(histo)
    en_img = np.zeros_like(img)
    for i in range(len(histo)):
        eq_histo[i] = int((L - 1) * np.sum(histo[0:i]))
    print_histogram(eq_histo, name="eq_"+str(index), title="Equalized Histogram")
    '''enhance image as well:'''
    for x_pixel in range(img.shape[0]):
        for y_pixel in range(img.shape[1]):
            pixel_val = int(img[x_pixel, y_pixel])
            en_img[x_pixel, y_pixel] = eq_histo[pixel_val]
    '''creating new histogram'''
    hist_img, _ = generate_histogram(en_img, False)
    print_img(img=en_img, histo_new=hist_img, histo_old=histo, index=str(index), L=L)
    return eq_histo

def find_value_target(val, target_arr):
    key = np.where(target_arr == val)[0]

    if len(key) == 0:
        key = find_value_target(val+1, target_arr)
        if len(key) == 0:
            key = find_value_target(val-1, target_arr)
    vvv = key[0]
    return vvv

def print_img(img, histo_new, histo_old, index, L):
    dpi = 80
    width = img.shape[0]
    height = img.shape[1]
    if height > width:
        figsize = (img.shape[0]*4) / float(dpi), (height)/ float(dpi)
        fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [3, 1,1]}, figsize=figsize)
    else:
        figsize = (width) / float(dpi), (height*4) / float(dpi)
        fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1,1]}, figsize=figsize)

    fig.suptitle("Enhanced Image with L:" + str(L))
    axs[0].title.set_text("Enhanced Image")
    axs[0].imshow(img, vmin=np.amin(img), vmax=np.amax(img), cmap='gray')

    axs[1].title.set_text("Equalized histogram")
    axs[1].plot(histo_new, color='#f77f00')
    axs[1].bar(np.arange(len(histo_new)), histo_new, color='#003049')

    axs[2].title.set_text("Main histogram")
    axs[2].plot(histo_old, color='#ef476f')
    axs[2].bar(np.arange(len(histo_old)), histo_old, color='#b7b7a4')
    plt.tight_layout()
    plt.savefig("e" + index + str(L)+".pdf")
    plt.savefig("e" + index + str(L)+".png")

def match_histogram(inp_img, hist_input, e_hist_input, e_hist_target, _print=True):
    '''map from e_inp_hist to 'target_hist '''
    en_img = np.zeros_like(inp_img)
    tran_hist = np.zeros_like(e_hist_input)
    for i in range(len(e_hist_input)):
        tran_hist[i] = find_value_target(val=e_hist_input[i], target_arr=e_hist_target)
    print_histogram(tran_hist, name="trans_hist_", title="Transferred Histogram")
    '''enhance image as well:'''
    for x_pixel in range(inp_img.shape[0]):
        for y_pixel in range(inp_img.shape[1]):
            pixel_val = int(inp_img[x_pixel, y_pixel])
            en_img[x_pixel, y_pixel] = tran_hist[pixel_val]
    '''creating new histogram'''
    hist_img, _ = generate_histogram(en_img, print=False, index=3)
    # cv2.imwrite('here', en_img)
    print_img(img=en_img, histo_new=hist_img, histo_old=hist_input, index=str(3), L=L)

imga_dir = '/home/percv-d0/dyn/2023.1_BSpaper/datasets/klicense_dataset/images/train/Cars70_K98wn9355.png'
# Cars70_K98wn9355.png
imgb_dir = '/home/percv-d0/dyn/2023.1_BSpaper/datasets/klicense_dataset/realimg/IMG-7734.JPG'
imga = cv2.imread(imga_dir, cv2.COLOR_BGR2GRAY)
imgb = cv2.imread(imgb_dir,  cv2.COLOR_BGR2GRAY)

# imga = load_images(imga_dir)
print(imga)

L = 50
index = 0
gr_img_arr = []
gr_hist_arr = []
eq_hist_arr = []

hist_imga, gr_imga = generate_histogram(imga, True)
hist_imgb, gr_imgb = generate_histogram(imgb, True)
# gr_hist_arr.append(hist_imga)
# gr_img_arr.append(gr_img)
eq_hist_input = equalize_histogram(gr_imga, hist_imga, L)
eq_hist_target = equalize_histogram(gr_imgb, hist_imgb, L)

match_histogram(inp_img=gr_imga, hist_input=hist_imga, e_hist_input=eq_hist_input, e_hist_target=eq_hist_target)
# a_hist, a_histimg = generate_histogram(imga, 0)
# b_hist, b_histimg = generate_histogram(imgb, 0)

# 
# match_histogram(imga, a_hist, b_hist)