import face_recognition
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from scipy.spatial import distance as dist

def get_facial_similarity(reference_image, unknown_image):
    try:
        reference_encoding = face_recognition.face_encodings(reference_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        results = face_recognition.compare_faces([reference_encoding], unknown_encoding)
        return results[0]
    except IndexError:
        return False

def get_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=1, n_init=3)
    clt.fit(image)
    return clt.cluster_centers_[0]

def get_color_similarity(color1, color2):
    return dist.euclidean(color1, color2)

def get_texture_similarity(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    lbp1 = local_binary_pattern(image1_gray, 24, 8, method="uniform")
    lbp2 = local_binary_pattern(image2_gray, 24, 8, method="uniform")
    
    (hist1, _) = np.histogram(lbp1.ravel(), bins=np.arange(0, 27), range=(0, 26))
    (hist2, _) = np.histogram(lbp2.ravel(), bins=np.arange(0, 27), range=(0, 26))
    
    hist1 = hist1.astype("float32")
    hist1 /= (hist1.sum() + 1e-7)
    hist2 = hist2.astype("float32")
    hist2 /= (hist2.sum() + 1e-7)
    
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
