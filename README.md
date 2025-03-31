# README for RealData Code

## RealData_0: Original Image Preprocessing
This dataset consists of image information and bounding box annotation information. All images are stored in the directory **Images_all_8cls**, and their corresponding annotation information is recorded in the file **master_file_8cls.npy**.

## RealData_1: Generate Features and Corresponding Labels for Sub - images
- **Images_random_8cls**: This directory stores the original images of 8 classes that have been divided into training and test sets.
- **master_file_train_8cls.npy**: Contains the training set data of 8 classes and the resized bounding box information.
- **master_file_test_8cls.npy**: Contains the test set data of 8 classes and the resized bounding box information.

In this stage,  we generate sub-images and corresponding features using transfer learning techniques. According to the annotation information, the corresponding label $Y$ will be generated for each sub-image. The generated features and labels of the training set are saved in **512_traindata_8cls.npz**, while those of the test set are saved in **512_testdata_8cls.npz**.

## RealData_2: Model Coefficient Calculation
In this stage, the GMLE, PMLE, and SPMLE methods are used for calculation, and the results are stored in the **model_coeff**.

## RealData_3: Visualization
This part demonstrates how the images in the main body of the paper are drawn.
