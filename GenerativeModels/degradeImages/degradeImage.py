import numpy as np
import cv2
import scipy as sc
import skimage as ski

class degradeImage:

    def __init__(self, img):
        '''
        :param img: The 'img' parameter refers to the image to modify
        '''
        self.original = img
        self.img = img
        self.img = self.img.astype(np.float64)
        self.img -= self.img.min()
        self.img /= self.img.max()

    def impulseNoise(self, thres):
        '''
        A matrix of zeros with the dimensions of image is generated.
        to make sure there are no issues, the 'dtype' is specified to be the same as 'img'
        The zero matrix is refilled with randomly generated values that follow a
        uniformly distributed function with values of 0 and 255
        To create the noise, a threshold is applied to filter the amount of points.
        The 'thres' value represents the maximum value that a pixel can have. If this value is exceeded,
        it is then changed based on the next parameter, which is fixed (the functionality of this threshold 'thres'
        was configured to be inversed, this means, the bigger the 'thres' value, the bigger the noise).
        Then the noise is added to the image.
        :param thres: The 'thres' value represents the maximum value that a pixel can have. If this value is exceeded, it is then changed based on the next parameter, which is fixed wit the value 0.1.
        :return: it is returned the image with impulse noise
        '''

        imp_noise = np.zeros(self.img.shape, dtype=np.float64)
        cv2.randu(imp_noise, 0, 255)
        imp_noise = cv2.threshold(imp_noise, thres, 0.1, cv2.THRESH_BINARY_INV)[1]

        im_img = self.img + imp_noise
        im_img -= im_img.min()
        im_img /= im_img.max()

        return im_img

    def gaussianNoise(self,sigma):
        '''
        A matrix of zeros with the dimensions of image is generated.
        to make sure there are no issues, the 'dtype' is specified to be the same as image
        The zero matrix is now refilled with randomly generated values that follow the Gaussian distribution
        using the 'sigma' value to create the noise, Finally, the noise is added to image
        :param sigma: The 'sigma' parameter refers to the standard deviation used to generate the noise
        :return: it is returned the image with gaussian noise
        '''

        gauss_noise=np.zeros(self.img.shape,dtype=self.img.dtype)
        cv2.randn(gauss_noise,0,sigma)
        gauss_noise = (gauss_noise).astype(self.img.dtype)
        gauss_noise -= gauss_noise.min()
        gauss_noise /= gauss_noise.max()

        gn_img = self.img + gauss_noise
        gn_img -= gn_img.min()
        gn_img /= gn_img.max()

        return gn_img

    def speckleNoise(self, intensity):
        '''
        A matrix of random values with the dimensions of image is generated. Then the matrix is normalized.
        the image is multiplied by both the noise and the desired intensity,
        and the result is ultimately added to the image
        :param intensity: The 'intensity' parameter refers to the intensity of the noise

        :return: it is returned the image with speckle noise
        '''

        row,col = self.img.shape
        speck = np.random.randn(row,col)
        speck -= speck.min()
        speck /= speck.max()

        sp_img = self.img + self.img * speck * intensity
        sp_img -= sp_img.min()
        sp_img /= sp_img.max()

        return sp_img

    def poissonNoise(self, intensity):
        '''
        Poisson noise is pixel-dependent, so a matrix with random values
        following the Poisson distribution is generated,
        and each pixel is condition by the previus value in the image
        The noise can be too strong, which is why we reduce its intensity
        by multiplying it by values between 0 and 1
        :param intensity: The 'intensity' parameter refers to the intensity of the noise
        :return: it is returned the image with speckle noise
        '''

        poisson_noise = np.random.poisson(self.img)
        poisson_noise = poisson_noise.astype(np.float64)
        poisson_noise -= poisson_noise.min()
        poisson_noise /= poisson_noise.max()

        poisson_noise = poisson_noise * intensity
        poi_img = self.img - poisson_noise
        poi_img -= poi_img.min()
        poi_img /= poi_img.max()

        return poi_img


    def stripesNoise(self, intensity, frec):
        '''
        It is taken the row, col values of the 'img'
        Using the values of row and col two 1D arrays are created with values
        between 0 and 1 with the two previous arrays a mesh is created with a 2D form
        The grid of X now is passed to the funtion to modelate the stripes,
        then the stripes multiplied by the desire intensity and then added to 'img'

        :param intensity: The 'intensity' parameter refers to the intensity of the noise
        :param frec: The 'frec' parameter refers to the frequency of the sin function
        :return: it is returned the image with stripes noise
        '''

        row, col = self.img.shape
        x = np.linspace(0, 1, col)
        y = np.linspace(0, 1, row)
        [X, Y] = np.meshgrid(x, y)

        stripes = np.sin(frec * 2 * np.pi * X)
        stripes -= stripes.min()
        stripes /= stripes.max()

        st_img = self.img + stripes * self.img * intensity
        st_img -= st_img.min()
        st_img /= st_img.max()

        return st_img


    def lines(self, space,thickness,pha):
        '''
        To draw the lines, a list of arrays is initialized, starting with the first streak.
        To add the remaining streaks, a for loop is initiated, applying the formulas.
        The list is then converted into a NumPy array. OpenCV's 'polylines' function is used to draw the lines.
        The 'addWeighted' function it is applied to control the opacity of the lines.
        :param space: The 'space' parameter refers to the space betwen lines
        :param thickness: The 'thickness' parameter refers to the thickness of the lines
        :param pha: The 'pha' parameter refers to the alpha that controls the opacity of the image
        :return: it is returned the image with lines
        '''

        points = [[space, self.img.shape[0]], [space, 0]]

        for i in range(0, self.original.shape[0] + 4, space):
            if (i + space + space < self.img.shape[0]):
                n = np.random.uniform(low=1, high=10, size=None)
                points.append([i, 0])
                points.append([i + space, 0])
                points.append([i + space, self.original.shape[0]])
                points.append([i + space + space, self.original.shape[0]])
                points.append([i + space + space, 0])

            else:
                break

        points = np.array(points)
        img = self.original.copy()
        overlay = self.original.copy()

        im_lines = cv2.polylines(img, [points], False, (128, 128, 128), thickness)
        alpha = pha
        ln_img = cv2.addWeighted(overlay, alpha, im_lines, 1 - alpha, 0)
        ln_img -= ln_img.min()
        ln_img /= ln_img.max()

        return ln_img

    def convolutionG1(self, intensity):
        '''
        The function generates correlated noise, with a convolution between a g funtion and a gaussian kernel.
        :param intensity: The 'intensity' parameter refers to the intensity of the noise
        :return: it is returned the image with a convolutional noise
        '''
        hor, verti = self.img.shape[0], self.img.shape[1]
        g1 = np.zeros((hor, verti))
        x1 = np.linspace(-verti // 2, verti // 2, verti)
        x2 = np.linspace(-hor // 2, hor // 2, hor)
        gauss_noise = np.random.normal(loc=0.0, scale=0.05, size=self.img.shape)
        for i, y in enumerate(x2, start=0):
            for j, x in enumerate(x1, start=0):
                g1[i, j] = max(0, 1 - abs(y)) * max(0, 16 - abs(x))
        covG1 = sc.signal.convolve2d(g1, gauss_noise, mode='same')
        covG1 -= covG1.min()
        covG1 /= covG1.max()
        gn_covG1_img = self.img + covG1 * intensity
        gn_covG1_img -= gn_covG1_img.min()
        gn_covG1_img  /= gn_covG1_img.max()
        return gn_covG1_img

    def convolutionG2(self,intensity):
        '''
        The function generates correlated noise, with a convolution between a g funtion and a gaussian kernel.
        :param intensity: The 'intensity' parameter refers to the intensity of the noise
        :return: it is returned the image with a convolutional noise
        '''
        hor, verti = self.img.shape[0], self.img.shape[1]
        g2 = np.zeros((hor, verti))
        x1 = np.linspace(-verti // 2, verti // 2, verti)
        x2 = np.linspace(-hor // 2, hor // 2, hor)
        [X,Y] = np.meshgrid(x2,x1)
        gauss_noise= np.random.normal(loc=0.0, scale=0.05, size=self.img.shape)
        for i, y in enumerate(x2, start=0):
            for j, x in enumerate(x1, start=0):
                g2[i, j] = np.cos((((x)**2)+((y)**2))**(1/2))
        covG2 = sc.signal.convolve2d(g2, gauss_noise, mode='same')
        covG2 -= covG2.min()
        covG2 /= covG2.max()
        gn_covG2_img = self.img + covG2 * intensity
        gn_covG2_img -= gn_covG2_img.min()
        gn_covG2_img /= gn_covG2_img.max()
        return gn_covG2_img

    def convolutionG1V(self,intensity):
        '''
        The function generates correlated noise, with a convolution between a g funtion and a gaussian kernel.
        :param intensity:  The 'intensity' parameter refers to the intensity of the noise
        :return: it is returned the image with a convolutional noise
        '''
        hor, verti = self.img.shape[0], self.img.shape[1]
        g1v = np.zeros((hor, verti))
        x1 = np.linspace(-verti // 2, verti // 2, verti)
        x2 = np.linspace(-hor // 2, hor // 2, hor)
        gauss_noise = np.random.normal(loc=0.0, scale=0.05, size=self.img.shape)
        for i, y in enumerate(x2, start=0):
            for j, x in enumerate(x1, start=0):
                g1v[i, j] = max(0, 16 - abs(y)) * max(0, 1 - abs(x))
        covG1v = sc.signal.convolve2d(g1v,gauss_noise, mode="same")
        covG1v -= covG1v.min()
        covG1v /= covG1v.max()
        gn_covG1v_img = self.img + covG1v*intensity
        gn_covG1v_img -= gn_covG1v_img.min()
        gn_covG1v_img /= gn_covG1v_img.max()
        return gn_covG1v_img

    def convolutionG1I(self, intensity, thres):
        '''
        The function generates correlated noise, with a convolution between a g funtion and a impulse kernel.
        :param intensity: The 'intensity' parameter refers to the intensity of the noise
        :param thres: The 'thres' value represents the maximum value that a pixel can have. If this value is exceeded, it is then changed based on the next parameter, which is fixed wit the value 0.1.
        :return: it is returned the image with a convolutional noise
        '''
        hor, verti = self.img.shape[1], self.img.shape[0]
        g1I = np.zeros((hor, verti))

        x1 = np.linspace(-verti // 2, verti // 2, verti)
        x2 = np.linspace(-hor // 2, hor // 2, hor)
        for i, y in enumerate(x2, start=0):
            for j, x in enumerate(x1, start=0):
                g1I[i, j] = max(0, 16 - abs(y)) * max(0, 1 - abs(x))

        imp_noise = np.zeros(self.img.shape, dtype=np.float64)
        cv2.randu(imp_noise, 0, 255)
        imp_noise = cv2.threshold(imp_noise, thres, 0.1, cv2.THRESH_BINARY_INV)[1]
        covG1I = sc.signal.convolve2d(g1I, imp_noise, mode="same")
        covG1I -= covG1I.min()
        covG1I /= covG1I.max()
        covG1I_img = covG1I * self.img
        covG1I_img -= covG1I_img.min()
        covG1I_img /= covG1I_img.max()
        im_covG1I_img = covG1I_img * intensity + self.img
        im_covG1I_img -= im_covG1I_img.min()
        im_covG1I_img /= im_covG1I_img.max()
        return im_covG1I_img
    def gaussianBlur(self, sigma):
        """
        The function applies the blur through a kernel, which in this case is a small matrix,
        and a convolution is performed with the image to obtain the blurry image
        :param sigma: he 'sigma' parameter refers to the standart deviation which determines the blur
        :return: it is returned the image with gaussian blur
        """
        gn_blur_img = ski.filters.gaussian(
            self.img, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
        return gn_blur_img