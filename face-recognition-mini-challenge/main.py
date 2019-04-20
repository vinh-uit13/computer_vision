
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm
from confusion_matrix import plot_confusion_matrix, call_plt
import pickle
import cv2
import numpy as np
import os

modelVGG16 = VGG16(weights='imagenet', include_top=True)
modelVGG16 = Model(input=modelVGG16.input, output=modelVGG16.get_layer('fc2').output)

modelVGG19 = VGG19(weights='imagenet', include_top=True)
modelVGG19 = Model(input=modelVGG19.input, output=modelVGG19.get_layer('fc2').output)

from imgaug import augmenters as iaa

def image_augmentor(img, name_img):
    seqs = [
        iaa.Sequential([
            iaa.SaltAndPepper(p=0.03),
        ]),
        iaa.Sequential([
            iaa.Invert(0.1, per_channel=True),
            iaa.GaussianBlur(sigma=(0,2.0))
        ]),
        iaa.Sequential([
            iaa.Fliplr(0.5), 
            iaa.Invert(0.05, per_channel=True),
            iaa.GaussianBlur(sigma=(0,3.0)) 
        ]),
        iaa.Sequential([
            iaa.MotionBlur(k=5)
        ])
    ]
    step = 0
    for seq in seqs:
        images_aug = seq.augment_images(img)            
        imgPath = name_img + '_aug_' + str(step) + '.jpg'
        cv2.imwrite(imgPath, images_aug)
        step += 1

def data_augmentation(dataset):
    listName = os.listdir(dataset)
    for name in listName:
        imgList = os.listdir(os.path.join(dataset, name))
        print ('This is name: ', name)
        for imgName in imgList:
            path = os.path.join(name, imgName)
            imgPath = os.path.join(dataset, path)
            img = cv2.imread(imgPath)
            image_augmentor(img, imgPath)
            print (imgPath)

def extractVGG16(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return modelVGG16.predict(x).reshape((4096,1))

def extractVGG19(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return modelVGG19.predict(x).reshape((4096,1))

def saveDict(obj, pathFile):
    pickle_out = open(pathFile, "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()

def loadDict(pathFile):
    pickle_in = open(pathFile, "rb")
    return pickle.load(pickle_in)

def load_img(dataset):
    listName = os.listdir(dataset)
    list_features = []
    list_labels = []
    for name in listName:
        if name.endswith('.pkl'):
            continue
        imgList = os.listdir(os.path.join(dataset, name))
        print ('This is name: ', name)
        for imgName in imgList:
            path = os.path.join(name, imgName)
            imgPath = os.path.join(dataset, path)
            img = image.load_img(imgPath, target_size=(224, 224))
            list_labels.append(name)
            list_features.append(extractVGG19(img))
            print ('This is imgPath: ', imgPath)

    json = {
        'list_features': list_features,
        'list_labels': list_labels
    }
    saveDict(json, 'dataset.pkl')

def train_model(dataset_feature, method, title):
    json = loadDict(dataset_feature)

    X = [item.reshape(-1) for item in json['list_features']]
    y = json['list_labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

    if method == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X_train, y_train)
    elif method == 'svm':
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train) 

    predict_list = clf.predict(X_test)
    target_names = set(y_test)

    plot_confusion_matrix(y_test, predict_list, classes=target_names, title=title, normalize=True)    
    print ('TITLE: ', title)
    print classification_report(y_test, predict_list, target_names=target_names)
    

if __name__ == '__main__':
    # load_img('train')
    train_model('dataset_vgg16.pkl', 'knn', 'VGG16-KNN')
    train_model('dataset_vgg19.pkl', 'knn', 'VGG19-KNN')
    train_model('dataset_vgg16.pkl', 'svm', 'VGG16-SVM')
    train_model('dataset_vgg19.pkl', 'svm', 'VGG19-SVM')

    # data_augmentation('train')
    # image_augmentor(cv2.imread('elephant.jpg'))
