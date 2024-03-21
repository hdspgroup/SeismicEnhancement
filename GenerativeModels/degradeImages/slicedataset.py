
import numpy as np
import math

class sliceDataset:

    def __init__(self, data):
        '''
        :param data: The 'data' parameter refers to the dataset
        '''
        self.data = data
        data -= data.min()
        data /= data.max()

    def slice128(self, div):
        '''
        The function uses math operations to calculate the amount of patches of 128X128 px
        that can be extracted from the .npy dataset. If the amount of patches doesn't fit in the image,
        an overlap of the pixels needed will be applied.
        :param div: The 'div' parameter works to reduce the amount of patches
        by dividing it by the 'div' value. If you don't want to reduce it, use 1 as the 'div' value.

        :return: An array of patches of the dataset
        '''
        if len(self.data.shape) >= 3:
            patch_number = self.data.shape[1] / 128
            patch_number = math.ceil(patch_number)
            surplus1 = patch_number * 128 - self.data.shape[1]
            patch_numberX = self.data.shape[0] / 128
            patch_numberX = math.ceil(patch_numberX)
            surplus = patch_numberX * 128 - self.data.shape[0]
            overlap = math.floor(surplus / (patch_numberX - 1))
            if (surplus1 < patch_number):
                overlap1 = surplus1
            if (surplus1 >= patch_number):
                overlap1 = math.floor(surplus1 / (patch_number - 1))
            patches = []
            c = overlap1
            for i in range(int(self.data.shape[2] / div)):
                usable_data = self.data[..., i]
                if (surplus1 < patch_number):
                    for i in range(patch_number):
                        if (patch_number - i <= overlap1):
                            c -= 1

                            sample = usable_data[::, i * 128 - 1 * (overlap1 - c):i * 128 + 128 - 1 * (overlap1 - c)]
                        else:
                            sample = usable_data[::, i * 128:i * 128 + 128]

                        for j in range(patch_numberX):
                            if (j == 0):
                                samplex = sample[j * 128:j * 128 + 128, ::]
                            else:
                                samplex = sample[j * 128 - overlap * j:j * 128 + 128 - overlap * j, ::]
                            if (j * 128 + 128 - overlap * j > sample.shape[0]):
                                dif = (j * 128 + 128 - overlap * j) - sample.shape[0]
                                samplex = sample[j * 128 - overlap * j - dif:j * 128 + 128 - overlap * j - dif, ::]

                            patches.append(samplex)
                if (surplus1 >= patch_number):
                    for i in range(patch_number):
                        if (i == 0):
                            sample = usable_data[::, i * 128:i * 128 + 128]
                        else:
                            sample = usable_data[::, i * 128 - overlap1 * i:i * 128 + 128 - overlap1 * i]
                        if (i * 128 + 128 - overlap1 * i > usable_data.shape[1]):
                            dif = (i * 128 + 128 - overlap1 * i) - usable_data.shape[1]
                            sample = usable_data[::, i * 128 - overlap1 * i - dif:i * 128 + 128 - overlap1 * i - dif]

                        for j in range(patch_numberX):
                            if (j == 0):
                                samplex = sample[j * 128:j * 128 + 128, ::]
                            else:
                                samplex = sample[j * 128 - overlap * j:j * 128 + 128 - overlap * j, ::]
                            if (j * 128 + 128 - overlap * j > sample.shape[0]):
                                dif = (j * 128 + 128 - overlap * j) - sample.shape[0]

                                if (dif == 0):
                                    samplex = sample[j * 128 - overlap * j:j * 128 + 128 - overlap * j, ::]
                                else:
                                    samplex = sample[j * 128 - overlap * j - dif:j * 128 + 128 - overlap * j - dif, ::]
                            patches.append(samplex)
        elif len(self.data.shape) == 2:
            patch_number = self.data.shape[1] / 128
            patch_number = math.ceil(patch_number)
            surplus1 = patch_number * 128 - self.data.shape[1]
            patch_numberX = self.data.shape[0] / 128
            patch_numberX = math.ceil(patch_numberX)
            surplus = patch_numberX * 128 - self.data.shape[0]
            overlap = math.floor(surplus / (patch_numberX - 1))
            if (surplus1 < patch_number):
                overlap1 = surplus1
            if (surplus1 >= patch_number):
                overlap1 = math.floor(surplus1 / (patch_number - 1))
            patches = []
            c = overlap1
            usable_data = self.data[...]

            if (surplus1 < patch_number):
                for i in range(patch_number):
                    if (patch_number - i <= overlap1):
                        c -= 1
                        sample = usable_data[::, i * 128 - 1 * (overlap1 - c):i * 128 + 128 - 1 * (overlap1 - c)]
                    else:
                        sample = usable_data[::, i * 128:i * 128 + 128]
                    for j in range(patch_numberX):
                        if (j == 0):
                            samplex = sample[j * 128:j * 128 + 128, ::]
                        else:
                            samplex = sample[j * 128 - overlap * j:j * 128 + 128 - overlap * j, ::]
                        if (j * 128 + 128 - overlap * j > sample.shape[0]):
                            dif = (j * 128 + 128 - overlap * j) - sample.shape[0]
                            samplex = sample[j * 128 - overlap * j - dif:j * 128 + 128 - overlap * j - dif, ::]

                        patches.append(samplex)
            if (surplus1 >= patch_number):
                for i in range(patch_number):
                    if (i == 0):
                        sample = usable_data[::, i * 128:i * 128 + 128]
                    else:
                        sample = usable_data[::, i * 128 - overlap1 * i:i * 128 + 128 - overlap1 * i]
                    if (i * 128 + 128 - overlap1 * i > usable_data.shape[1]):
                        dif = (i * 128 + 128 - overlap1 * i) - usable_data.shape[1]
                        sample = usable_data[::, i * 128 - overlap1 * i - dif:i * 128 + 128 - overlap1 * i - dif]
                    for j in range(patch_numberX):
                        if (j == 0):
                            samplex = sample[j * 128:j * 128 + 128, ::]
                        else:
                            samplex = sample[j * 128 - overlap * j:j * 128 + 128 - overlap * j, ::]
                        if (j * 128 + 128 - overlap * j > sample.shape[0]):
                            dif = (j * 128 + 128 - overlap * j) - sample.shape[0]
                            if (dif == 0):
                                samplex = sample[j * 128 - overlap * j:j * 128 + 128 - overlap * j, ::]
                            else:
                                samplex = sample[j * 128 - overlap * j - dif:j * 128 + 128 - overlap * j - dif, ::]
                        patches.append(samplex)

        return patches