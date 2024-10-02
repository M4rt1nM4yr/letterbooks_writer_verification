import numpy as np
import cv2 as cv
import torch
from torchvision import transforms
from skimage.filters import threshold_sauvola, threshold_niblack, threshold_otsu, rank
from skimage.morphology import disk
from sklearn.preprocessing import normalize
from timeit import default_timer as time
from PIL import Image, ImageOps

def hellinger_kernel(data, split=50000):
    out = []
    for i in range(0,round(len(data)/split+0.5)):
        out.append(normalize(data[split*i:split*(i+1)], norm='l1', axis=1).astype(np.float32))
    out = np.concatenate(out)
    return np.sqrt(out)

class Descriptor():
    def __init__(
            self,
            binarizer="otsu",
            sampler="sift",
            descriptor="sift",
            hellinger_norm=True,
            rm_duplicates=True,
    ):
        self.binarizer = binarizer
        self.descriptor = descriptor
        self.sampler = sampler
        self.hellinger_norm = hellinger_norm
        self.rm_duplicates = rm_duplicates
        self.toPIL = transforms.ToPILImage()

        if binarizer == "otsu":
            self.binarize = bin_otsu
        else:
            raise ValueError(f"Unknown binarizer {binarizer}")

        if sampler == "sift":
            self.sample = extract_sift_keypoints
        else:
            raise ValueError(f"Unknown sampler {sampler}")

        if descriptor == "sift":
            self.descriptor_fn = extract_sift_descriptors
        else:
            raise ValueError(f"Unknown descriptor {descriptor}")

    def __call__(self,X):
        """
        :param x: should be a numpy array or a torch tensor or a list of numpy arrays or torch tensors
        :return:
        """
        if not isinstance(X,list):
            X = [X]
        all_features = list()
        for x in X:
            if isinstance(x,torch.Tensor):
                x = np.asarray(self.toPIL(x)).copy()
            elif isinstance(x,Image.Image):
                x = np.asarray(x).copy()
            assert isinstance(x,np.ndarray)
            x_bin = self.binarize(x)
            # show_cv_image(x_bin)
            points = self.sample(x_bin)
            if len(points) == 0:
                continue
            for p in points:
                p.angle = 0
            if self.rm_duplicates:
                kpt_coord = np.zeros( x.shape, dtype=int)
                for e,kp in enumerate(points):
                    tt = kpt_coord[ int(kp.pt[1]), int(kp.pt[0]) ]
                    # keep kpt w. strongest response
                    if tt == 0 or (tt != 0 and kp.response > points[tt].response):
                        kpt_coord[ int(kp.pt[1]), int(kp.pt[0]) ] = e
                rel_coords = np.nonzero(kpt_coord.ravel())
                ind = kpt_coord.ravel()[rel_coords]
                points = np.array(points)[ ind ]
            features = self.descriptor_fn(img=x_bin, points=points)
            all_features.append(features)
        features = np.concatenate(all_features,axis=0)
        if self.hellinger_norm:
            features = hellinger_kernel(features)
        return features

def extract_sift_keypoints(x):
    sift = cv.SIFT_create(
        contrastThreshold=0.01,
        edgeThreshold=40,
    )
    keypoints = sift.detect(x,None)
    return keypoints

def bin_otsu(x):
    # Apply Otsu's binarization
    _, binary_image = cv.threshold(x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Find connected components
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary_image, 8, cv.CV_32S)

    # Define size threshold
    size_threshold = 30  # Adjust this value according to your needs

    # Create an output image
    output_image = np.zeros_like(binary_image)

    # Loop through components and keep only the large ones
    for label in range(1, num_labels):
        if stats[label, cv.CC_STAT_AREA] >= size_threshold:
            output_image[labels == label] = 255

    return output_image

def extract_sift_descriptors(img, points):
    # show_cv_image(img)
    sift = cv.SIFT_create()

    # keypoint_visualization = cv.drawKeypoints(img, points, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # # Display the image with keypoints
    # cv.imshow("Keypoints", keypoint_visualization)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    keypoints, descriptors = sift.compute(img,points)

    return descriptors
