import os
import warnings

# Suppress TensorFlow and ONNX Runtime logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ORT_LOGGING_LEVEL'] = '3'
# Suppress specific InsightFace alignment warning
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", message=".*SimilarityTransform.from_estimate.*")

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
import os

# Initialize InsightFace with Buffalo_L
# Using CPUExecutionProvider since CUDA dlls are missing on the system
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# Pre-prepare with large size for small faces in drone footage
app.prepare(ctx_id=0, det_size=(1280, 1280))

def get_faces(image):
    """
    Detects all faces in the image using Buffalo_L and returns face objects with embeddings.
    """
    if image is None:
        return []
    try:
        # InsightFace expects BGR image (OpenCV default)
        faces = app.get(image)
        return faces
    except Exception as e:
        print(f"InsightFace Detection Error: {e}")
        return []

def get_face_embedding(image):
    """
    Extracts a 512-dimensional embedding using InsightFace ArcFace.
    """
    try:
        faces = app.get(image)
        if len(faces) > 0:
            # Sort by face size to get the most prominent face
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            return faces[0].embedding
        return None
    except Exception as e:
        print(f"InsightFace Embedding Error: {e}")
        return None

def get_cosine_similarity(feat1, feat2):
    """
    Computes cosine similarity between two 512-dim embeddings.
    """
    if feat1 is None or feat2 is None:
        return 0.0
    try:
        # Cosine similarity = 1 - cosine distance
        return 1 - dist.cosine(feat1, feat2)
    except Exception as e:
        print(f"Cosine Similarity Error: {e}")
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
