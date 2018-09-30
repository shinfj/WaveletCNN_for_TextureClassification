# coding:utf-8
import numpy as np


# mean + diff
def WaveletTransformAxisY(img):
    row, col = img.shape[:2]
    size = row / 2

    img_even = img[1::2]
    img_odd = img[0::2]
    if len(img_even) != len(img_odd):
        img_odd = img_odd[:-1]
    # c: mean (low-frequency), d: diff (high-frequency)
    c = (img_even + img_odd) / 2.
    d = abs(img_odd - img_even)

    return size, c, d

def WaveletTransformLLAxisY(img):
    row, col = img.shape[:2]
    img_even = img[1::2]
    img_odd = img[0::2]
    if len(img_even) != len(img_odd):
        img_odd = img_odd[:-1]
    # c: mean (low-frequency), d: diff (high-frequency)
    c = (img_even + img_odd) / 2.

    return c

def WaveletTransformAxisX(img):
    tmp = np.fliplr(img.T)
    size, dst_L, dst_H = WaveletTransformAxisY(tmp)
    dst_L = np.flipud(dst_L.T)
    dst_H = np.flipud(dst_H.T)
    return size, dst_L, dst_H

def WaveletTransformLLAxisX(img):
    tmp = np.fliplr(img.T)
    dst_L = WaveletTransformLLAxisY(tmp)
    dst_L = np.flipud(dst_L.T)
    return dst_L

def WaveletTransform(img, n=1):
    row, col = img.shape[:2]
    roi = img[0:row,0:col]
    wavelets = {}
    for i in range(0,n):
        y_size, wavelet_L, wavelet_H = WaveletTransformAxisY(roi)
        x_size, wavelet_LL, wavelet_LH = WaveletTransformAxisX(wavelet_L)
        wavelets["LL_"+str(i+1)] = wavelet_LL
        wavelets["LH_"+str(i+1)] = wavelet_LH
        x_size, wavelet_HL, wavelet_HH = WaveletTransformAxisX(wavelet_H)
        wavelets["HL_"+str(i+1)] = wavelet_HL
        wavelets["HH_"+str(i+1)] = wavelet_HH
        roi = wavelet_LL
    return wavelets
