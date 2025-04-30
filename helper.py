import copy
import os
import sys

import cv2
import numpy as np

from pathlib import Path
from PIL import Image
from scipy import ndimage

from treeofshapes import TreeOfShapes
from skimage import io
from skimage.metrics import structural_similarity as ssim
import csv
import torch

from torchvision import transforms
import lpips
import threading

sys.setrecursionlimit(5000)

def imshow(image, click_event=None, cmap=None, figsize=None, vmin=None, vmax=None):
    """
    Show an image at true scale
    """
    import matplotlib.pyplot as plt
    dpi = 80
    margin = 0.5  # (5% of the width/height of the figure...)
    if figsize is None:
        h, w = image.shape[:2]
    else:
        h, w = figsize

    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * w / dpi, (1 + margin) * h / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(image, interpolation='none', cmap=cmap, vmin=vmin, vmax=vmax)

    plt.axis('off')
    plt.show()

    return fig, ax

def generate_pointwise_mean(image_path, target_folder='pointwise_mean_output/'):
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    k_size = 3
    step = 2
    kernel = (k_size, k_size)

    Path(target_folder).mkdir(parents=True, exist_ok=True)
    for i in range(0,10):
        blurred = cv2.blur(im, kernel)
        Image.fromarray(blurred).save(target_folder + '/conv_mean' + str(k_size) + '.png')
        k_size = k_size + step
        kernel = (k_size, k_size)

def generate_pointwise_median(image_path, target_folder='pointwise_median_output/'):
    image = Image.open(image_path)

    k_size = 3
    step = 2
    kernel = (k_size, k_size)

    Path(target_folder).mkdir(parents=True, exist_ok=True)
    for i in range(0,10):
        median = ndimage.median_filter(image, size=kernel)
        cv2.imwrite(target_folder + '/conv_median' + str(k_size) + '.png', median)
        k_size = k_size + step
        kernel = (k_size, k_size)



def generate_pointwise_median_consecutive(image_path, nb_it, target_folder='pointwise_median_cons_output/'):
    image = Image.open(image_path)
    ksize = 3
    kernel = (ksize, ksize)

    Path(target_folder).mkdir(parents=True, exist_ok=True)
    for i in range(0, nb_it):
        image = ndimage.median_filter(image, size=kernel)
        cv2.imwrite(target_folder + '/conv_median_x' + str(i + 1) + '.png', image)

def generate_pointwise_mean_consecutive(image_path, nb_it, target_folder='pointwise_median_cons_output/'):
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    ksize = 3
    kernel = (ksize, ksize)

    Path(target_folder).mkdir(parents=True, exist_ok=True)
    for i in range(0, nb_it):
        im = cv2.blur(im, kernel)
        Image.fromarray(im).save(target_folder + '/conv_mean_x' + str(i + 1) + '.png')

def generate_bilateral_filtering(image_path, target_folder='bilateral_output'):
    image = cv2.imread(image_path)
    d = 5
    sigmacolorspace = 10
    Path(target_folder).mkdir(parents=True, exist_ok=True)
    dincrement = 2
    colorincrement = 5
    for i in range(0,30):
        filtered = cv2.bilateralFilter(image, d, sigmacolorspace, sigmacolorspace)
        gray_filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(target_folder + '/bilateral_d' + str(d) + '_c' + str(sigmacolorspace) + '.png', gray_filtered)
        sigmacolorspace += colorincrement
        d += dincrement

def incremental_tos_filter(image_path, filter_type, output_fp, nb_filter=10):
    im = io.imread(image_path)
    tos = TreeOfShapes(im)

    Path(output_fp).mkdir(parents=True, exist_ok=True)
    if filter_type == 'pp':
        pp = 2
        for i in range(0, nb_filter):
            tos_pp = copy.deepcopy(tos)
            tos_pp.filter_tree_proper_part_bottom_up(pp, to_parent=True)
            res = tos_pp.reconstruct_image()
            res.save(output_fp + '/tos_pp_' + str(pp) + '.png')
            pp += 1
    if filter_type == 'area':
        area = 2
        for i in range(0, nb_filter):
            tos_area = copy.deepcopy(tos)
            tos_area.filter_tree_area_bottom_up(area)
            res = tos_area.reconstruct_image()
            res.save(output_fp + '/tos_area_' + str(area) + '.png')
            area += 1
    elif filter_type == 'mean':
        for i in range(0, nb_filter):
            tos.filter_tree_mean_v2()
            im = tos.reconstruct_image()
            im.save(output_fp + '/tos_mean_x' + str(i + 1) + '.png')

    elif filter_type == 'median':
        for i in range(0, nb_filter):
            tos.filter_tree_median()
            im = tos.reconstruct_image()
            im.save(output_fp + '/tos_median_x' + str(i + 1) + '.png')

def generate_tos_stats(filepath):
    with open(filepath + 'tos_nodes.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Header
        header = ["Image", "Nb nodes", "Nb leaves", "Nb unary non leaf", "Nb one to parent",
                 "Nb leaves %", "Nb unary non leaf %", "Nb one to parent %"]
        writer.writerow(header)

        for imname in os.listdir(filepath):
            if imname.endswith(".png") or imname.endswith(".jpg"):
                im = io.imread(filepath + imname)
                tos = TreeOfShapes(im)
                stats_map = tos.stats_tree()
                row = [imname,
                       str(stats_map['nb_nodes']),
                       str(stats_map['nb_leaves']),
                       str(stats_map['nb_non_leaf_unary']),
                       str(stats_map['nb_one_to_parent']),
                       str(stats_map['nb_leaves_pct']),
                       str(stats_map['nb_non_leaf_unary_pct']),
                       str(stats_map['nb_one_to_parent_pct'])]
                writer.writerow(row)

def run_exp_pipeline(filepath='dataset/', output_fp='output/', padding=False):
    orig_fp = filepath
    first = True

    for imname in os.listdir(orig_fp):
        if imname.endswith(".png") or imname.endswith(".jpg"):
            print("Treating", imname, '...')
            if padding:
                print("Padding on.", end=' ')
                image = io.imread(orig_fp + imname)
                image = np.pad(image, 1, mode="constant", constant_values=0)

                if first:
                    filepath = filepath + 'padded/'
                    print("Creating folder", filepath + '.')
                    Path(filepath).mkdir(parents=True, exist_ok=True)
                    first = False
                print("Saving padding to", filepath + imname + '.')
                Image.fromarray(image).save(filepath + imname)

            folder_names = []
            threads = []
            # run bilateral filter
            #t1 = threading.Thread(target=__run_bilateral, args=(filepath, folder_names, imname, output_fp,))
            #threads.append(t1)

            # run area
            #t2 = threading.Thread(target=run_tos_area, args=(filepath, folder_names, imname, output_fp))
            #threads.append(t2)

            #t3 = threading.Thread(target=__run_tos_proper_part, args=(filepath, folder_names, imname, output_fp,))
            #threads.append(t3)
            print("tos mean")
            # run tos mean
            t4 = threading.Thread(target=__run_tos_mean, args=(filepath, folder_names, imname, output_fp, 300))
            threads.append(t4)

            # run tos median
            t5 = threading.Thread(target=__run_tos_median, args=(filepath, folder_names, imname, output_fp, 300))
            threads.append(t5)

            # run conv mean
            #t6 = threading.Thread(target=run_conv_mean, args=(filepath, folder_names, imname, output_fp,))
            #threads.append(t6)

            # run conv median
            #t7 = threading.Thread(target=run_conv_median, args=(filepath, folder_names, imname, output_fp,))
            #threads.append(t7)

            # run consecutive mean
            t8 = threading.Thread(target=run_conv_mean_cons, args=(filepath, folder_names, imname, output_fp, 300))
            threads.append(t8)

            # run consecutive median
            t9 = threading.Thread(target=run_conv_median_cons, args=(filepath, folder_names, imname, output_fp, 300))
            threads.append(t9)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # SSIM, PSNR, LPIPS
            for fn in folder_names:
                generate_metrics(fn, filepath + imname)
                generate_tos_stats(fn)
    generate_tos_stats(filepath)


def run_conv_median(filepath, folder_names, imname, output_fp):
    fn = output_fp + '/pointwise_median_cons/' + imname + '/'
    folder_names.append(fn)
    generate_pointwise_median(filepath + imname, fn)

def run_conv_median_cons(filepath, folder_names, imname, output_fp, nb_it):
    fn = output_fp + '/pointwise_median_cons/' + imname + '/'
    folder_names.append(fn)
    generate_pointwise_median_consecutive(filepath + imname, nb_it, fn)

def run_conv_mean_cons(filepath, folder_names, imname, output_fp, nb_it):
    fn = output_fp + '/pointwise_mean/' + imname + '/'
    folder_names.append(fn)
    generate_pointwise_mean_consecutive(filepath + imname, nb_it, fn)

def run_conv_mean(filepath, folder_names, imname, output_fp):
    fn = output_fp + '/pointwise_mean/' + imname + '/'
    folder_names.append(fn)
    generate_pointwise_mean(filepath + imname, fn)

def __run_tos_median(filepath, folder_names, imname, output_fp, nb_it):
    fn = output_fp + '/tos_median/' + imname + '/'
    folder_names.append(fn)
    incremental_tos_filter(filepath + imname, 'median', fn, nb_filter=nb_it)

def __run_tos_mean(filepath, folder_names, imname, output_fp, nb_it):
    fn = output_fp + '/tos_mean/' + imname + '/'
    folder_names.append(fn)
    incremental_tos_filter(filepath + imname, 'mean', fn, nb_filter=nb_it)

def __run_tos_proper_part(filepath, folder_names, imname, output_fp):
    fn = output_fp + '/proper_part/' + imname + '/'
    folder_names.append(fn)
    incremental_tos_filter(filepath + imname, 'pp', fn, nb_filter=50)

def run_tos_area(filepath, folder_names, imname, output_fp):
    fn = output_fp + '/area/' + imname + '/'
    folder_names.append(fn)
    incremental_tos_filter(filepath + imname, 'area', fn, nb_filter=50)


def __run_bilateral(filepath, folder_names, imname, output_fp):
    fn = output_fp + '/bilateral/' + imname + '/'
    folder_names.append(fn)
    generate_bilateral_filtering(filepath + imname, fn)


def image_to_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return preprocess(image).unsqueeze(0)

def compute_lpips(ref_path, image_path):
    ref = image_to_tensor(ref_path)
    img = image_to_tensor(image_path)
    with torch.no_grad():
        loss_fn = lpips.LPIPS(net='alex', verbose=False)
        d = loss_fn.forward(ref, img)
        return d.item()

def generate_metrics(filepath, original):
    im_ref = cv2.imread(original)
    im_ref = np.squeeze(im_ref)

    with open(filepath + 'metric.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Write header
        header = ["Image", 'SSIM', 'PSNR', 'LPIPS', 'MSE']
        writer.writerow(header)

        for imname in os.listdir(filepath):
            if imname.endswith(".png"):
                impath = filepath + imname
                image = cv2.imread(impath)

                ssim_score = ssim(im_ref, image, channel_axis=2)
                psnr = cv2.PSNR(im_ref, image)
                lpips = compute_lpips(original, impath)
                mse = get_mse(im_ref, image)

                row = [imname, str(ssim_score), str(psnr), str(lpips), str(mse)]
                writer.writerow(row)

def compute_mse_folder(filepath, orig_fp):
    im_ref = cv2.imread(orig_fp)
    im_ref = np.squeeze(im_ref)
    with open(filepath + 'mse.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        header = ["Image", 'MSE']
        writer.writerow(header)
        for imname in os.listdir(filepath):
            if imname.endswith(".png") or imname.endswith(".jpg"):
                impath = filepath + imname
                image = cv2.imread(impath)
                mse = get_mse(im_ref, image)
                row = [imname, str(mse)]
                writer.writerow(row)

def get_mse(img1, img2):
    img1 = np.squeeze(img1)
    h, w = img1.shape[:2]
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff ** 2)
    mse = err / (float(h * w))
    return mse

