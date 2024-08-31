import os
import cv2 as cv
import numpy as np
import pandas as pd
from glob import glob
from pydicom import dcmread
from tqdm import tqdm

BASEDIR = os.path.dirname(__file__)


def show_dicom(dir_dicom: str, save=False) -> dict:
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
            
            path_idname = os.path.join(basepath, classname, idname)
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


if __name__ == '__main__':
    # D:/Annisa/AD_AXIAL_T2_STAR/ADNI/002_S_5018/Axial_T2-Star/2012-11-08_07_53_28.0/I346239
    # show_dicom('D:/Annisa/AD_AXIAL_T2_STAR/ADNI/002_S_5018/Axial_T2-Star/2012-11-08_07_53_28.0/I346239')
    
    list_info = []
    for dirpath, dirnames, filenames in os.walk('D:/Annisa/dataset_alzheimer'):
        # print('Dir:', dirpath)
        # for filename in filenames:
        #     print('---------', filename)
        # print()
        
        paths = glob(f"{dirpath}/*.dcm")
        if not paths:
            continue
        
        print(dirpath)

        info = show_dicom(dirpath, save=True)
        info.update(
            dict(zip(('Class', 'Source', 'Subject', 'Description', 'AcgDate', 'ID'), dirpath.split('\\')[1:]))
        )
        list_info.append(info)

        print()
        
        # if len(list_info) == 3:
        #     break
        
    # df = pd.DataFrame(list_info)
    # df.to_csv(os.path.join(BASEDIR, 'result', 'show_dicom.csv'), index=False)
    
    # ds = dcmread('D:/Annisa/AD_AXIAL_T2_STAR/ADNI/006_S_4153/Axial_T2-Star/2011-08-03_08_12_01.0/I248520/ADNI_006_S_4153_MR_Axial_T2-Star__br_raw_20110803163318472_1_S117305_I248520.dcm')
    # # ds = dcmread('D:/Annisa/AD_AXIAL_T2_STAR/ADNI/002_S_5018/Axial_T2-Star/2013-11-18_10_41_00.0/I398679/ADNI_002_S_5018_MR_Axial_T2-Star__br_raw_20131118125912373_8_S206235_I398679.dcm')
    
    # print(ds.pixel_array.shape)
    
    # for i in range(ds.pixel_array.shape[0]):
    #     img = cv.normalize(ds.pixel_array[i], None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    #     cv.imshow('ds', img)
    #     if cv.waitKey(30) & 0xFF == ord('q'):
    #         break

    # cv.destroyAllWindows()
    
    pass
