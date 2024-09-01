import os
import cv2 as cv
import numpy as np
import pandas as pd
from glob import glob
from pydicom import dcmread
from tqdm import tqdm

from alzheption.functions import (
    calculate_background_uniformity,
    calculate_brightness,
    calculate_contrast,
    calculate_shadow,
    calculate_sharpness,
    calculate_specularity,
    calculate_unique_intensity_levels,
)

BASEDIR = os.path.dirname(__file__)


def show_dicom(dir_dicom: str, dir_jpg='dataset_jpg', save=False) -> dict:
    info = {
        'n_dicom': 0,
        'shape': None,
    }
    
    paths = glob(f"{dir_dicom}/*.dcm")
    info.update({'n_dicom': len(paths)})
    
    # --- sort dicom files based on frames sequence.
    paths = sorted(paths, key=lambda x: int(x.split('_')[-3]))
    
    ii = 0
    for path in tqdm(paths, desc='Showing dicom'):
        if save:
            basepath, classname, _, _, _, _, idname, _ = path.split('\\')
            
            basepath = os.path.join(*basepath.split('/')[:-1])
            classname = classname.split('_')[0]
            idname = idname.lower()
            
            path_idname = os.path.join(basepath, dir_jpg, classname, idname)
            os.makedirs(path_idname, exist_ok=True)
        
        ds = dcmread(path)
        if len(ds.pixel_array.shape) > 2:
            for i in tqdm(range(ds.pixel_array.shape[0]), 'Showing dicom frames'):
                img = cv.normalize(ds.pixel_array[i], None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
                
                if save:
                    path_img = os.path.join(path_idname, f"{i}.jpg")
                    cv.imwrite(path_img, img)
                
                cv.imshow('ds', img)
                if cv.waitKey(30) & 0xFF == ord('q'):
                    break

            cv.destroyAllWindows()
            
            continue
        
        # --- normalize dicom pixel array np.uint16 (0-65535) to np.uint8 (0-255) only for display dicom with opencv.
        img = cv.normalize(ds.pixel_array, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        
        if save:
            path_img = os.path.join(path_idname, f"{ii}.jpg")
            cv.imwrite(path_img, img)
        
        ii += 1
        
        cv.imshow('ds', img)
        if cv.waitKey(30) & 0xFF == ord('q'):
            break
    
    info.update({'shape': ds.pixel_array.shape})

    cv.destroyAllWindows()
    
    return info


def calculate_image_attributes(path: str) -> dict:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    
    return {
        'Shape': str(img.shape),
        'Sharpness': calculate_sharpness(img),
        'Brightness': calculate_brightness(img),
        'Contrast': calculate_contrast(img),
        'UIL': calculate_unique_intensity_levels(img),
        'Shadow': calculate_shadow(img),
        'Specularity': calculate_specularity(img),
        'BU': calculate_background_uniformity(img),
    }


if __name__ == '__main__':
    # # --- extract dicom to jpg
    # list_info = []
    # for dirpath, dirnames, filenames in os.walk('D:/Annisa/dataset_alzheimer'):
    #     # print('Dir:', dirpath)
    #     # for filename in filenames:
    #     #     print('---------', filename)
    #     # print()
        
    #     paths = glob(f"{dirpath}/*.dcm")
    #     if not paths:
    #         continue
        
    #     print(dirpath)

    #     info = show_dicom(dirpath, save=True)
    #     info.update(
    #         dict(zip(('Class', 'Source', 'Subject', 'Description', 'AcgDate', 'ID'), dirpath.split('\\')[1:]))
    #     )
    #     list_info.append(info)

    #     print()
        
    # # df = pd.DataFrame(list_info)
    # # df.to_csv(os.path.join(BASEDIR, 'result', 'show_dicom.csv'), index=False)
    
    # --- extract image (jpg) attributes
    list_attribute = []
    for dirpath, dirnames, filenames in os.walk('D:/Annisa/dataset_jpg'):
        paths = glob(f"{dirpath}/*.jpg")
        if not paths:
            continue
        
        cls, idx = dirpath.split('\\')[-2:]
        # print(dirpath, cls, idx)
        
        for path in tqdm(paths, desc=f"[{cls}, {idx}] Extracting"):
            name = path.split('\\')[-1].split('.')[0]
            
            path_jpg = os.path.join(dirpath, path)
            
            attributes = calculate_image_attributes(path_jpg)
            attributes.update({'Class': cls, 'ID': idx, 'Name': name})
            
            list_attribute.append(attributes)
    
    df = pd.DataFrame(list_attribute)
    df.to_csv(os.path.join(BASEDIR, 'result', 'img_attributes.csv'), index=False)
    
    pass
