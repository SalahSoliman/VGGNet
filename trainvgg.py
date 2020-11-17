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

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", help="plots output loc")
args = vars(ap.parse_args())

print("[INFO] Loading the CIFAR10 dataset...")

((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]

print("[INFO] Compling the model...")

print(np.shape(trainY))
input_h = trainX.shape[1]
input_W = trainX.shape[2]
input_d = trainX.shape[3]
input_classes = trainY.shape[1]
model = MiniVGGNet.build(input_h, input_W, input_d, input_classes)
opt = SGD(0.05)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

print("[INFO] Training the network...")
epochs = 40
H = model.fit(trainX, trainY, batch_size=64, epochs=epochs, verbose=1, shuffle=True)

print("[INFO] Evaluating Network...")

predictions = model.predict(testX, batch_size=64)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))
model.save("model.hdf5")

plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, epochs), H.history["loss"], label="Loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="Val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="Accuracy")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], labels="Val_accuracy")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
