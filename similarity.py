import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
import os

# Set environment variable to suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_facial_similarity(reference_image, unknown_image):
    """
    Calculates facial similarity using DeepFace with Cosine Similarity.
    Returns a score between 0 and 1, where 1 is a perfect match.
    """
    try:
        # DeepFace.verify returns a dictionary with 'distance' and 'similarity_metric'
        # Default model is VGG-Face, metric is 'cosine'
        result = DeepFace.verify(
            img1_path=reference_image,
            img2_path=unknown_image,
            model_name='VGG-Face',
            distance_metric='cosine',
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # Cosine distance is 0 for identical vectors, 1 for orthogonal.
        # We convert distance to a similarity score (1 - distance)
        distance = result['distance']
        similarity = max(0, 1 - distance)
        return similarity
    except Exception as e:
        print(f"DeepFace Error: {e}")
        return 0.0

def get_dominant_color(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = KMeans(n_clusters=1, n_init=3)
        clt.fit(image)
        return clt.cluster_centers_[0]
    except Exception as e:
        print(f"Color Analysis Error: {e}")
        return np.array([0, 0, 0])

def get_color_similarity(color1, color2):
    return dist.euclidean(color1, color2)

def get_texture_similarity(image1, image2):
    try:
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
    except Exception as e:
        print(f"Texture Analysis Error: {e}")
        return 0.0
