import os
import shutil
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


def calculate_image_attributes(path: str, normalize=False) -> dict:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    
    return {
        'Shape': str(img.shape),
        'Sharpness': calculate_sharpness(img, normalize=normalize),
        'Brightness': calculate_brightness(img, normalize=normalize),
        'Contrast': calculate_contrast(img, normalize=normalize),
        'UIL': calculate_unique_intensity_levels(img, normalize=normalize),
        'Shadow': calculate_shadow(img),
        'Specularity': calculate_specularity(img),
        'BU': calculate_background_uniformity(img),
    }


def re_sort() -> None:
    df = pd.read_csv('alzheption/result/img_attributes_normalized.csv')
    df = df.sort_values(by=['Class', 'ID', 'Name'])
    
    for cls, idx in tqdm(df[['Class', 'ID']].drop_duplicates().values, desc='Re-Sorting'):
        name_jpg = os.listdir(os.path.join('D:', 'Annisa', 'dataset_jpg', cls, idx))[0]
        
        if 'x' in name_jpg:
            df_x = df[df.ID == idx].sort_values(by=['Name'])
            ids_x = df_x.index.tolist()
            val_x = sorted(df_x.Name.tolist(), reverse=True)
            df.loc[ids_x, 'Name'] = val_x
    else:
        df.to_csv('alzheption/result/img_attributes_normalized_resort.csv', index=False)
    
    return None


def pick_dataset_by_sort(src_path: str, dst_path, k=5) -> None:
    os.makedirs(dst_path, exist_ok=True)
    
    i = 0
    for dirpath, dirnames, filenames in os.walk('D:/Annisa/dataset_jpg'):
        paths = glob(f"{dirpath}/*.jpg")
        s_idx = int((16/44)*len(paths))
        paths = sorted(paths, key=lambda x: int(os.path.split(x)[-1].split('.')[0].split('-')[0]))[s_idx:s_idx + 5]
        
        if not paths:
            continue
        
        for path in tqdm(paths, desc='Copying...'):
            path_new = path.replace(src_path, dst_path)
            
            directory = os.path.split(path_new)[0]
            os.makedirs(directory, exist_ok=True)
            
            shutil.copy(path, path_new)
        
        i += 1


def pick_dataset_by_brightness(
    path_img_attributes_data='alzheption/result/img_attributes_normalized_resort.csv',
    feature_by='Brightness', 
    ascending=True,
    base_path='D:/Annisa', 
    src_dir='dataset_jpg', 
    dst_dir='dataset_jpg_picked'
) -> None:
    df = pd.read_csv(path_img_attributes_data)
    df = df.sort_values(by=['Class', 'ID', feature_by], ascending=ascending)
    df = df.groupby('ID').tail()
    
    src_dir = os.path.join(base_path, src_dir)
    dst_dir = os.path.join(base_path, dst_dir)
    for cls, idx, name in tqdm(df[['Class', 'ID', 'Name']].values, desc='Building dataset'):
        path = os.path.join(src_dir, cls, idx, f"{name}.jpg")
        if not os.path.exists(path):
            path = os.path.join(src_dir, cls, idx, f"{name}-x.jpg")
        
        dst_dir_idx = os.path.join(dst_dir, cls, idx)
        os.makedirs(dst_dir_idx, exist_ok=True)
        
        path_idx = os.path.join(dst_dir_idx, f"{name}.jpg")
        
        shutil.copy(path, path_idx)


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
    
    # # --- extract image (jpg) attributes
    # list_attribute = []
    # for dirpath, dirnames, filenames in os.walk('D:/Annisa/dataset_jpg'):
    #     paths = glob(f"{dirpath}/*.jpg")
    #     if not paths:
    #         continue
        
    #     cls, idx = dirpath.split('\\')[-2:]
    #     # print(dirpath, cls, idx)
        
    #     for path in tqdm(paths, desc=f"[{cls}, {idx}] Extracting"):
    #         name = path.split('\\')[-1].split('.')[0]
            
    #         path_jpg = os.path.join(dirpath, path)
            
    #         attributes = calculate_image_attributes(path_jpg)
    #         attributes.update({'Class': cls, 'ID': idx, 'Name': name})
            
    #         list_attribute.append(attributes)
    
    # df = pd.DataFrame(list_attribute)
    # df.to_csv(os.path.join(BASEDIR, 'result', 'img_attributes.csv'), index=False)
    
    
    # # --- sort image (jpg) by Sharpness and UIL
    # df = pd.read_csv('alzheption/result/img_attributes_normalized.csv')
    # df = df.sort_values(by=['Class', 'ID', 'Name'])
    
    # for idx in df.ID.unique():
    #     df_idx = df[df.ID == idx]
        
    #     s_sn = df_idx.iloc[0, 1]
    #     s_uil = df_idx.iloc[0, 4]
    #     e_sn = df_idx.iloc[-1, 1]
    #     e_uil = df_idx.iloc[-1, 4]
        
    #     if s_sn < e_sn and s_uil > e_uil:
    #         cls, idx = df_idx.iloc[0, -3:-1].tolist()
    #         print(cls, idx)
            
    #         for e, i in enumerate(df_idx.index.tolist()[::-1]):
    #             cls = df_idx.loc[i, 'Class']
    #             idx = df_idx.loc[i, 'ID']
    #             name = df_idx.loc[i, 'Name']
                
    #             path_jpg = os.path.join('D:', 'Annisa', 'dataset_jpg', cls, idx, f"{name}.jpg")
    #             path_new = os.path.join('D:', 'Annisa', 'dataset_jpg', cls, idx, f"{e}-x.jpg")
                
    #             # img = cv.imread(path_jpg, cv.IMREAD_GRAYSCALE)
    #             # cv.imshow('img', img)
    #             # if cv.waitKey(10) & 0xFF == ord('q'):
    #             #     break
                
    #             os.rename(path_jpg, path_new)
    
    # --- re-sort goes here
    
    # --- pick dataset by brightness
    # pick_dataset_by_brightness(feature_by='Brightness', ascending=True, dst_dir='dataset_jpg_brightness')
    # pick_dataset_by_brightness(feature_by='BU', ascending=False, dst_dir='dataset_jpg_bu')
    
    # # analyze above: maybe we can use BU
    # # update: based on BU, some images return no object
    # df = pd.read_csv('alzheption/result/img_attributes_normalized_resort.csv')
    # df = df.sort_values(by=['Class', 'ID', 'Name'])
    # print(df[df.ID == 'i306070'].sort_values(by=['Brightness']).tail())
    
    # print(df[df.ID == 'i306070'].sort_values(by=['BU']).head())
    
    pass
