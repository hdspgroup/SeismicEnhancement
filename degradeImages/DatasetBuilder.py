# This is a sample Python script.
from degradeImage import degradeImage
from slicedataset import sliceDataset

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os





def degradeBatch(batch):
    """
    The function degrade clean images randomly with a minimun of 3 kinds of damage and
    return an array of degraded images.
    :param batch: The "batch" parameter refers to the array of images that will be damaged
    :return: The function return an array of images
    """
    degradedBatch = []
    for img in batch:
        gauss = False
        poi = False
        stripe = False
        impulse = False
        speck = False
        covG1 = False
        covG2 = False
        covG1I = False
        covG1V = False
        gaussBlur = False
        numNoise = np.random.randint(3, 9)
        for i in range(numNoise):
            typeNoise = np.random.randint(0, 9)
            if (typeNoise == 0 and gauss == False):
                gauss = True
                sigma = np.random.random()
                img = degradeImage(img).gaussianNoise(sigma)
            elif (typeNoise == 1 and poi == False):
                poi = True
                intensity = np.random.uniform(0.2, 0.4)
                img = degradeImage(img).poissonNoise(intensity)
            elif (typeNoise == 2 and stripe == False):
                stripe = True
                intensity = np.random.uniform(0.15, 0.3)
                frec = np.random.randint(6, 15)
                img = degradeImage(img).stripesNoise(intensity, frec)
            elif (typeNoise == 3 and impulse == False):
                impulse = True
                thres = np.random.randint(15, 30)
                img = degradeImage(img).impulseNoise(thres)
            elif (typeNoise == 4 and speck == False):
                speck = True
                intensity = np.random.uniform(0.2, 0.4)
                img = degradeImage(img).speckleNoise(intensity)
            elif (typeNoise == 5 and covG1 == False):
                covG1 = True
                intensity = np.random.uniform(0.2, 0.7)
                img = degradeImage(img).convolutionG1(intensity)
            elif (typeNoise == 6 and covG2 == False):
                covG2 = True
                intensity = np.random.uniform(0.2, 0.7)
                img = degradeImage(img).convolutionG2(intensity)
            elif (typeNoise == 7 and covG1V == False):
                covG1V = True
                intensity = np.random.uniform(0.1, 0.5)
                img = degradeImage(img).convolutionG1V(intensity)
            elif (typeNoise == 8 and covG1I == False):
                covG1I = True
                intensity = np.random.uniform(0.1, 0.3)
                img = degradeImage(img).convolutionG1I(intensity,15)
            elif (typeNoise == 9 and gaussBlur == False):
                gaussBlur = True
                sigma = np.random.random()
                img = degradeImage(img).gaussianBlur(sigma)
        degradedBatch.append(img)

    return degradedBatch

def degradeBatchMatrix(batch):
    """
    The function degrade images and return a matrix. The first position of every row is the original image,
    and each of the following positions represents a type of degradation that affects the original image
    :param batch: The "batch" parameter refers to the array of images that will be damaged
    :return: The function return a matrix with degradations
    """
    degradedBatch = np.zeros((len(batch), 13), dtype=object)
    print(1)
    for i in range(degradedBatch.shape[0]):
        degradedBatch[i][0] = batch[i]
        degradedBatch[i][1] = degradeImage(batch[i]).impulseNoise(30)
        degradedBatch[i][2] = degradeImage(batch[i]).gaussianNoise(1)
        degradedBatch[i][3] = degradeImage(batch[i]).poissonNoise(0.3)
        degradedBatch[i][4] = degradeImage(batch[i]).speckleNoise(0.3)
        degradedBatch[i][5] = degradeImage(batch[i]).convolutionG1(0.4)
        degradedBatch[i][6] = degradeImage(batch[i]).convolutionG2(0.6)
        degradedBatch[i][7] = degradeImage(batch[i]).convolutionG1V(0.5)
        degradedBatch[i][8] = degradeImage(batch[i]).convolutionG1I(0.4, 30)
        degradedBatch[i][9] = degradeImage(batch[i]).gaussianBlur(1)
        degradedBatch[i][10] = degradeImage(degradedBatch[i][5]).convolutionG2(0.4)
        degradedBatch[i][11] = degradeImage(degradedBatch[i][10]).gaussianBlur(1)
        degradedBatch[i][12] = degradeImage(degradedBatch[i][11]).convolutionG1I(0.3,20)
    return degradedBatch



if __name__ == '__main__':

    data_folder = "C:/Users/juter/Downloads/R_3.npy"
    data = np.load(data_folder)
    data = data.astype(np.float64)
    data -= data.min()
    data /= data.max()
    print(len(data.shape))
    batch = []
    batch = sliceDataset(data).slice128(1)

    
    for i in range(1000):
        batch[i] = np.rot90(batch[i])
        batch.append(np.rot90(data[..., i]))

    dataset = degradeBatchMatrix(batch)
    np.save(os.path.join("C:/Users/juter/Documents/seismic sgy/npy", 'degradedDataset.npy'), dataset)

    print(len(batch))
    #plt.imshow(dataset[0], cmap="seismic")
    #plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
