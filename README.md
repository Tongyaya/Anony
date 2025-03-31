# Pairwise Maximum Likelihood For Multi-Class Logistic Regression Model With Multiple Rare Classes

- Dependencies
Python == 3.7
torch == 1.3.1

- Make sure you have all the dependencies listed in `requirements.txt` installed
- Install them using `pip install -r requirements.txt`

- Simulation

Simu1.1 to Simu1.3 reproduce the results of Figure 1 in the manuscipt, comparing the GMLE and PMLE methods. Simu2 replicates the results of Table 1 in the manuscipt, comparing the PMLE and SPMLE methods. Simu3 contains the code specifically designed for visualizing key data and results. Simu4 is the supplementary experiment, providing additional empirical evidence to validate our research.

- RealData

## Dataset: **split_data**
Due to the anonymity requirements and the file size limitations imposed by GitHub, we can only upload the dataset after splitting and compressing it. Once the rebuttal process concludes, we will upload the dataset to Google Drive for easy access.

## RealData_0: Original Image Preprocessing
This TikTok Screenshots (TTS) dataset contains a total of 2,559 screenshots of size 720 $\times$ 1,280 randomly taken from TikTok live streams sponsored by different Audi dealers in China. All images are stored in the directory **Images_all_8cls**, and their corresponding annotation information is recorded in the file **master_file_8cls.npy**.

## RealData_1: Generate Features and Corresponding Labels for Sub - images
- **Images_random_8cls**: This directory stores the original images of 8 classes that have been divided into training and test sets.
- **master_file_train_8cls.npy**: Contains the training set data of 8 classes and the resized bounding box information.
- **master_file_test_8cls.npy**: Contains the test set data of 8 classes and the resized bounding box information.

In this step,  we generate sub-images and corresponding features using transfer learning techniques. According to the annotation information, the corresponding label $Y$ will be generated for each sub-image. The features and labels of the training set are saved in **512_traindata_8cls.npz**, while those of the test set are saved in **512_testdata_8cls.npz**.

## RealData_2: Model Coefficient Calculation
In this step, the GMLE, PMLE, and SPMLE methods are used for calculation. The computed results are stored in the **model_coef**.

## RealData_3: Visualization
This part demonstrates the process of creating the images presented in the manuscript.

## RealData_4: Empirical Comparison with Baseline Methods
We have included the following methods for comparison on the TTS dataset: the focal loss (FL) of Lin et al. (2017), the class-balanced loss (CBL) of Cui et al. (2019), the cost sensitive loss (CSL) and random downsampling (RDS) of Fernández et al. (2018). All methods are optimized according to the suggestions of the original papers.

### References
1. Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class - balanced loss based on effective number of samples. *CVPR*.
2. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *ICCV*.
3. Song, Y., & Zou, H. (2024). Minimax optimal rates with heavily imbalanced binary data. *IEEE TIT*.
4. Fernández, A., García, S., Galar, M., Prati, R. C., Krawczyk, B., & Herrera, F. (2018). *Learning from imbalanced data sets*. Springer. 
