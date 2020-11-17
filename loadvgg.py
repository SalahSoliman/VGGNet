import numpy as np
import matplotlib
matplotlib.use("agg")
from minivggnet import MiniVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt 
import argparse
from keras.models import load_model
import pickle
import cv2


print("[INFO] Loading the CIFAR10 dataset...")

((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]

model = load_model("model.hdf5")

preds = model.predict(testX, batch_size=32).argmax(axis=1)

print("[INFO]", testY[1].argmax())

# print(np.shape(testX[0]))
for i in np.arange(0, 20):
    print(labelNames[preds[i]], "    ", labelNames[testY[i].argmax()])
    cv2.imshow("Prediction", testX[i])
    cv2.waitKey(0)
