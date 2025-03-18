# Alzheption

Alzheimer classification using Inception-V3 towards 6 classess. Repair BibTeX with [Flaming Tempura](https://flamingtempura.github.io/)

## 🎯 TODOs

- [x] Create dataset on kaggle
- [x] Pick 5 images per subject
- [x] Sort images in each subject to specific order
- [x] Extract dicom to jpg
- [x] Download T2-Star dataset

## 📖 Documentation

- [Notebook](https://www.kaggle.com/code/hudafn/alzheption)
- [Dataset](https://www.kaggle.com/datasets/hudafn/alzheption-dataset)

## Question

1. CV accuracy hit 85% when add hflip augmentation with original transformation. But, its decrease when use hflip and rotate augmentation. Is rotate decrease the accuracy? or we need more neurons while we increase the augmentation?
    a. Accuracy has linear relation with increase of neurons, but it's very small.
2. After we add new augmentation again (increase the dataset count also) then accuracy was decrease.
3. Shear failed.
