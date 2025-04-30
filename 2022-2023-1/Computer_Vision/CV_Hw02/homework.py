# -*- coding: utf-8 -*-
import pickle
from PIL import Image
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random

images = []
labels = []
IMAGE_SIZE = 200


# 按照指定图像大小调整尺寸
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    return cv2.resize(image, (height, width))


def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read_path(full_path)
        else:  # 文件
            if dir_item.endswith('.jpg') or dir_item.endswith('.JPG') or dir_item.endswith('.png'):
                image = cv2.imread(full_path)
                image = resize_image(image)
                images.append(image)
                labels.append(path_name)

    return images, labels


def load_dataset(path_name):
    images, labels = read_path(path_name)

    images = np.array(images)
    print(images.shape)
    category = []
    for i in labels:
        category.append(i.split('/')[-1])
    temp = list(set(category))
    dic = {}
    for i in range(len(temp)):
        dic[temp[i]] = i
    for i in range(len(category)):
        labels[i] = dic[category[i]]
    labels = np.array(labels)
    print(labels.shape)
    return images, labels


class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_lb = None

        # 测试集
        self.test_images = None
        self.test_lb = None

        # 数据集加载路径
        self.path_name = path_name

        # 当前库采用的维度顺序
        self.input_shape = None

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=3, nb_classes=102):
        # 加载数据集到内存
        images, labels = load_dataset(self.path_name)

        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3,
                                                                                random_state=random.randint(0, 100))

        # 输入图片数据时的顺序为：rows,cols,channels
        train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
        test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
        self.input_shape = (img_channels, img_rows, img_cols)
        '''
        print(test_images[0].shape)
        cv2.imwrite('test.jpg', test_images[0])
        assert False
        '''

        # 输出训练集、验证集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(test_images.shape[0], 'test samples')

        self.train_lb = train_labels
        self.test_lb = test_labels

        # 像素数据浮点化以便归一化
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')

        # 将其归一化,图像的各像素值归一化到0~1区间
        train_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.test_images = test_images


print('start data loading...')
data = Dataset('./caltech-101')
data.load()
print('Data Loading Finished.')


# TODO 利用SIFT从训练图像中提取特征
# 如果有需要，你也可以在pass之外的地方填写相关代码，请自便，下同。
# vec_dict 第i项： i为类别，对应的字典为所有属于该类的sift特征点的信息。注意：kp与des一一对应。
vec_dict = {i:{'kp':[], 'des':[]} for i in range(102)}

sift = cv2.SIFT_create()
for i in range(data.train_images.shape[0]):
    tep = cv2.normalize(data.train_images[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    kp_vector, des_vector = sift.detectAndCompute(tep, None)

    # TODO 1
    # write the SIFT feature into the dictionary vec_dict
    vec_dict[data.train_lb[i]]['kp'].extend(kp_vector)
    vec_dict[data.train_lb[i]]['des'].extend(des_vector)

    '''
    # visualization of the SIFT feature extracted
    img = cv2.drawKeypoints(tep, kp_vector, tep, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints.jpg', img)
    raise ValueError("Have a glimpse into the SIFT feature extracted.")
    '''

    ########
print('SIFT Feature Extraction for Each Category Finished.')


# 统计最少特征点的类别
bneck_value = float("inf")
for i in range(102):
    if len(vec_dict[i]['kp']) < bneck_value:
        bneck_value = len(vec_dict[i]['kp'])

for i in range(102):
    kp_list = vec_dict[i]['kp'] = sorted((vec_dict[i]['kp']),
                                         key=lambda x: x.response,
                                         reverse=True)


# TODO 为每个类别选择同样多的特征点用于聚类。特征点个数bneck_value

vec_list = vec_dict[0]['des'][0:bneck_value]
for i in range(1, 102):

    # TODO 2
    vec_list = np.vstack((vec_list, vec_dict[i]['des'][0:bneck_value]))
    ########

vec_list = np.float64(vec_list)
print(vec_list.shape)

# TODO 对提取出的特征点使用Kmeans聚类，设定合适的聚类中心个数

# TODO 3
from sklearn.cluster import KMeans
N_clusters = 102
print('-----------------------------------------------------')
print('Start K-means Clustering...')

kmeans = KMeans(n_clusters=N_clusters, n_init = 5).fit(vec_list)

y_predict = kmeans.predict(vec_list)

# visualization of the K-means clustering result
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# vec_pca = pca.fit_transform(vec_list)
# print(pca.explained_variance_ratio_)
# plt.scatter(vec_pca[:, 0], vec_pca[:, 1], c=y_predict)
# plt.savefig('kmeans_solution_{}.pdf'.format(N_clusters))

print('K-means Clustering finished.')

########


# TODO 利用直方图统计每张图像中的特征点所属聚类中心的个数，将直方图归一化后便得到图像的特征向量。
print('-----------------------------------------------------')
print('Start generating feature vector for each image...')
num_images = data.train_images.shape[0]
hist_vector = np.zeros((num_images, N_clusters))
log_interval = num_images // 10
for i in range(num_images):
    tep = cv2.normalize(data.train_images[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # TODO 4
    kp_vector, des_vector = sift.detectAndCompute(tep, None)
    pred = kmeans.predict(np.float64(des_vector))
    hist_vector[i] = np.histogram(pred, bins=N_clusters, range=(0, N_clusters), density=True)[0]
    if (i % log_interval == 0):
        print("Feature Vector Extraction : {:.2f}% \t [{}/{}] Done.".format(
                            100.*i/float(num_images), i, num_images))

    ########

print('Feature Vector Extraction finished.')


# 使用SVM构建分类器
# 你可以自行构建分类器，也可以使用SVM
print('-----------------------------------------------------')
print('Start training SVM classifier...')
from sklearn import svm
classifier = svm.SVC(probability=True, kernel='rbf')
classifier.fit(hist_vector, data.train_lb)
print('SVM classifier training finished.')


# TODO 构建测试集并计算模型准确率
print('=====================================================')
print('Start testing...')
num_test_images = data.test_images.shape[0]
hist_test_vector = np.zeros((num_test_images, N_clusters))
for i in range(num_test_images):
    tep = cv2.normalize(data.test_images[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # TODO 5
    # Extract the SIFT feature of the test image and put them into corresponding clusters (bags).
    kp_vector, des_vector = sift.detectAndCompute(tep, None)
    pred = kmeans.predict(np.array(des_vector, dtype=np.float64))
    hist_test_vector[i] = np.histogram(pred, bins=N_clusters, range=(0, N_clusters), density=True)[0]
    
    #######
print('Feature Vector Extraction finished.\n')

# Real output
y_pred = classifier.predict(hist_test_vector)

# Accuracy
acc = classifier.predict(hist_test_vector)-data.test_lb
tep = len(acc[acc==0])
print('Accuracy', tep/len(data.test_lb))