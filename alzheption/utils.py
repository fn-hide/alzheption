import os
import cv2 as cv
import numpy as np
from glob import glob
from pydicom import dcmread
from tqdm import tqdm


def show_dicom(dir_dicom: str):
    paths = glob(f"{dir_dicom}/*.dcm")
    
    # --- sort dicom files based on frames sequence.
    paths = sorted(paths, key=lambda x: int(x.split('_')[-3]))
    
    for path in tqdm(paths, desc='Showing dicom'):
        filename = path.split('\\')
        
        ds = dcmread(path)
        
        # --- normalize dicom pixel array np.uint16 (0-65535) to np.uint8 (0-255) only for display dicom with opencv.
        img = cv.normalize(ds.pixel_array, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        cv.imshow('ds', img)
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    # D:/Ayang/AD_AXIAL_T2_STAR/ADNI/002_S_5018/Axial_T2-Star/2012-11-08_07_53_28.0/I346239
    # show_dicom('D:/Ayang/AD_AXIAL_T2_STAR/ADNI/002_S_5018/Axial_T2-Star/2012-11-08_07_53_28.0/I346239')
    
    for dirpath, dirnames, filenames in os.walk('D:/Ayang/AD_AXIAL_T2_STAR'):
        # print('Dir:', dirpath)
        # for filename in filenames:
        #     print('---------', filename)
        # print()
        
        paths = glob(f"{dirpath}/*.dcm")
        if not paths:
            continue
        
        print(dirpath)
        show_dicom(dirpath)
        print()
    
    pass
