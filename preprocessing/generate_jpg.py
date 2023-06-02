import cv2
import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops
from PIL import Image


def crop_img(img_read_re, img_read_le):
    otsu = sitk.OtsuThresholdImageFilter()
    otsu_image = otsu.Execute(img_read_re)
    otsu_array = sitk.GetArrayFromImage(otsu_image)

    invert_otsu = (np.ones(otsu_array.shape ) -otsu_array).astype(np.uint8)
    (contours ,_) = cv2.findContours(invert_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    max_ctr = contours[0]
    for ctr in contours:
        if cv2.contourArea(ctr) > cv2.contourArea(max_ctr):
            max_ctr= ctr
    img_temp = np.zeros(invert_otsu.shape)
    polygon = ctr
    cv2.fillPoly( img_temp, [max_ctr], [1] )
    otsu_array = np.ones(invert_otsu.shape ) -img_temp

    temp_img_re = sitk.GetArrayFromImage(img_read_re)

    temp_img_le_original = sitk.GetArrayFromImage(img_read_le)

    temp_img_re =(np.ones((img_read_re.GetSize()[1], img_read_re.GetSize()[0])) - otsu_array) * temp_img_re
    temp_img_le = (np.ones((img_read_le.GetSize()[1], img_read_le.GetSize()[0])) - otsu_array) * temp_img_le_original
    props = regionprops(np.array(temp_img_re > 0, np.uint8))
    r0, c0, r1, c1 = props[0].bbox
    temp_img_re = temp_img_re[r0:r1, c0:c1]

    temp_img_re = pre_processing_for_img(temp_img_re)

    temp_img_le = temp_img_le[r0:r1, c0:c1]

    temp_img_le = pre_processing_for_img(temp_img_le)

    return temp_img_re, temp_img_le


def resample_intensities(orig_img, bin_nr=256):
    v_count = 0
    img_list = []
    filtered = orig_img.copy()
    if np.min(orig_img.flatten()) < 0:
        filtered += np.min(orig_img.flatten())
    resampled = np.zeros_like(filtered)
    max_val_img = np.max(filtered.flatten())
    min_val_img = np.min(filtered.flatten())
    step = (max_val_img - min_val_img) / bin_nr

    for st in np.arange(step + min_val_img, max_val_img + step, step):
        resampled[(filtered <= st) & (filtered >= st - step)] = v_count
        v_count += 1

    return np.array(resampled, dtype=np.uint16)

def pre_processing_for_img(img):
    temp_img = img.copy()
    low_thr = np.quantile(temp_img[temp_img>0], 0.01)
    high_thr = np.quantile(temp_img[temp_img>0], 0.99)
    temp_img[temp_img<low_thr] = low_thr
    temp_img[temp_img>high_thr] = high_thr
    if len(np.unique(temp_img[temp_img>0])) > 256:
            temp_img_sampled = resample_intensities(temp_img[temp_img>0])
            temp_img[temp_img>0] = temp_img_sampled
    else:
        new_img =(temp_img-np.min(temp_img))/(np.max(temp_img)-np.min(temp_img))
        temp_img = (new_img*255).astype(np.uint8)
    return temp_img

def load_patient(path_input_recombined,path_input_low_energy):
    img_read_re = sitk.ReadImage(path_input_recombined)
    img_read_le = sitk.ReadImage(path_input_low_energy)
    img_re, img_le = crop_img(img_read_re, img_read_le)
    return img_le,img_re

def preprocessing(path_input_low_energy,path_input_recombined):
    clahe = cv2.createCLAHE(clipLimit =2.5, tileGridSize=(16,16))
    clahe_recombined_2 = cv2.createCLAHE(clipLimit =1.0, tileGridSize=(16,16))

    img_le, img_re = load_patient(path_input_recombined,path_input_low_energy)
    new_img_le =(img_le-np.min(img_le))/(np.max(img_le)-np.min(img_le))
    im_le = (new_img_le*255).astype(np.uint8)

    x_le = clahe.apply(im_le).astype(np.uint8)
    new_img_re =(img_re-np.min(img_re))/(np.max(img_re)-np.min(img_re))
    new_img_re = (new_img_re*255).astype(np.uint8)

    x_re = clahe.apply(new_img_re).astype(np.uint8)
    x_re_2 = clahe_recombined_2.apply(new_img_re).astype(np.uint8)

    temp_im_le = Image.fromarray(x_le)
    temp_im_re = Image.fromarray(x_re)
    temp_im_re_2 = Image.fromarray(x_re_2)
    merged_img =Image.fromarray( cv2.merge((x_le,x_re,x_re_2)) )
    img_rgb = merged_img.convert("RGB")
    img_bgr = np.asarray(img_rgb)[:, :, ::-1].copy()
    return img_bgr

