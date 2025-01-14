import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm
import json
import os
import cv2

import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import csv
from skimage.feature import local_binary_pattern, hog
from sklearn.cluster import KMeans
# Fonction pour calculer l'histogramme normalisé des couleurs
# Fonction pour calculer l'histogramme normalisé des couleurs
def calculate_normalized_histogram(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de lire l'image à {image_path}. Vérifiez le chemin.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    histSize = 256
    hist_r = cv2.calcHist([image], [0], None, [histSize], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [histSize], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [histSize], [0, 256])

    # Normalisation des histogrammes
    cv2.normalize(hist_r, hist_r, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_g, hist_g, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist_r.flatten(), hist_g.flatten(), hist_b.flatten()


# Fonction pour calculer la matrice de co-occurrence
def calculate_co_occurrence_matrix(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossible de lire l'image à {image_path}. Vérifiez le chemin.")

    num_gray_levels = 256
    distance = 1

    co_matrix = np.zeros((num_gray_levels, num_gray_levels))

    for i in range(image.shape[0] - distance):
        for j in range(image.shape[1] - distance):
            gray_level1 = image[i, j]
            gray_level2 = image[i + distance, j + distance]
            co_matrix[gray_level1, gray_level2] += 1

    if co_matrix.sum() > 0:
        co_matrix /= co_matrix.sum()

    contrast = np.sum((np.arange(num_gray_levels)[:, None] - np.arange(num_gray_levels)) ** 2 * co_matrix)
    dissimilarity = np.sum(np.abs(np.arange(num_gray_levels)[:, None] - np.arange(num_gray_levels)) * co_matrix)
    homogeneity = np.sum(co_matrix / (1 + np.abs(np.arange(num_gray_levels)[:, None] - np.arange(num_gray_levels))))
    energy = np.sum(co_matrix ** 2)

    return contrast, dissimilarity, homogeneity, energy


# Fonction pour calculer les descripteurs HOG (Histogram of Oriented Gradients)
def calculate_hog(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossible de lire l'image à {image_path}. Vérifiez le chemin.")

    # Calcul du HOG
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return fd


# Fonction pour calculer le LBP (Local Binary Pattern)
def calculate_lbp(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossible de lire l'image à {image_path}. Vérifiez le chemin.")

    # Paramètres LBP
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')

    # Calcul de l'histogramme LBP
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= lbp_hist.sum()  # Normalisation

    return lbp_hist


# Fonction pour calculer les moments de Hu (cinq descripteurs supplémentaires)
def calculate_hu_moments(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossible de lire l'image à {image_path}. Vérifiez le chemin.")

    # Seuil de l'image pour obtenir une image binaire
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calcul des moments de Hu
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()

    return hu_moments
# Function to calculate Gabor filter features
def calculate_gabor_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image at {image_path}. Check the path.")

    kernels = []
    gabor_features = []

    for theta in range(4):  # 0, 45, 90, 135 degrees
        theta = theta * np.pi / 4
        kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)

    for kernel in kernels:
        filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        gabor_features.append(np.mean(filtered_image))
        gabor_features.append(np.std(filtered_image))

    return np.array(gabor_features)

# Function to calculate dominant colors
def calculate_dominant_colors(image_path, k=5):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image at {image_path}. Check the path.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(image)
    dominant_colors = kmeans.cluster_centers_.flatten()

    return dominant_colors

class BayesianRelevanceFeedback:
    def __init__(self, database, image_ids):
        """
        Initialize the Bayesian Relevance Feedback system.

        Parameters:
        - database (dict): Dictionary of image paths and their descriptors.
        - image_ids (list): List of image paths corresponding to the database.
        """
        self.database = database  # Dictionary of descriptors
        self.image_ids = image_ids  # List of image paths
        self.weights = np.ones(len(next(iter(database.values()))))  # Equal initial weights
        self.feedback_received = False  # Flag to track feedback status

    def calculate_query_vector(self, query_path):
    # Calculate and normalize histogram features
     hist_r, hist_g, hist_b = calculate_normalized_histogram(query_path)
     hist = np.concatenate((hist_r, hist_g, hist_b))
     hist = hist / np.linalg.norm(hist)  # Normalize histogram features
     all_features = []
     all_features.extend(hist)
    
    # Calculate and normalize co-occurrence matrix features
     contrast, dissimilarity, homogeneity, energy = calculate_co_occurrence_matrix(query_path)
     co_occurrence = np.array([contrast, dissimilarity, homogeneity, energy])
     co_occurrence = co_occurrence / np.linalg.norm(co_occurrence)  # Normalize co-occurrence features

     all_features.extend(co_occurrence)
    
    # Calculate and normalize HOG features
     avg_hog = np.mean(calculate_hog(query_path))
    
     avg_hog = avg_hog / np.linalg.norm(avg_hog)  # Normalize HOG features
     all_features.append(avg_hog)
    
    # Calculate and normalize LBP features
     avg_lbp = calculate_lbp(query_path) / np.linalg.norm(calculate_lbp(query_path))
     all_features.extend(avg_lbp)
    
    # Calculate and normalize Hu Moments
     avg_hu = calculate_hu_moments(query_path) / np.linalg.norm(calculate_hu_moments(query_path))
     all_features.extend(avg_hu)
    
    # Calculate and normalize Gabor features
     avg_gabor = calculate_gabor_features(query_path) / np.linalg.norm(calculate_gabor_features(query_path))
     all_features.extend(avg_gabor)
    
    # Calculate and normalize dominant colors
     avg_dominant_colors = calculate_dominant_colors(query_path) / np.linalg.norm(calculate_dominant_colors(query_path))
     all_features.extend(avg_dominant_colors)
    
     return np.array(all_features)

    def initial_search(self, query_vector):
        """Perform the initial similarity search based on the query image."""
        print("Performing initial search.")
        similarities = []
        for image_id in self.image_ids:
            descriptor = np.array(self.database[image_id])
            similarity = cosine_similarity(query_vector.reshape(1, -1), descriptor.reshape(1, -1))[0, 0]
            similarities.append((image_id, similarity))
            sorted_images = sorted(similarities, key=lambda x: x[1], reverse=True)
        return sorted_images[:20]

    def update_feedback(self, query_vector, relevant_ids, irrelevant_ids):
        """Update feature weights based on user feedback."""
        # Extract relevant and irrelevant vectors
        relevant_vectors = np.array([
            np.array(self.database[img_id]) for img_id in relevant_ids
        ])
        irrelevant_vectors = np.array([
            np.array(self.database[img_id]) for img_id in irrelevant_ids
        ])
        

        # Compute mean and variance for relevant and irrelevant groups
        relevant_mean = np.mean(relevant_vectors, axis=0) if relevant_vectors.size else query_vector
        irrelevant_mean = np.mean(irrelevant_vectors, axis=0) if irrelevant_vectors.size else query_vector

        # Add a small epsilon to avoid division by zero
        relevant_var = np.var(relevant_vectors, axis=0) + 1e-6 if relevant_vectors.size else 1.0
        irrelevant_var = np.var(irrelevant_vectors, axis=0) + 1e-6 if irrelevant_vectors.size else 1.0

        # Update weights using Bayesian formula
        self.weights = np.log(
            (norm.pdf(query_vector, relevant_mean, np.sqrt(relevant_var)) + 1e-6) /
            (norm.pdf(query_vector, irrelevant_mean, np.sqrt(irrelevant_var)) + 1e-6)
        )
        self.weights = np.nan_to_num(self.weights, nan=0.0, posinf=0.0, neginf=0.0)
        self.feedback_received = True  # Mark feedback as received
        print("Weights updated.")
 
    def refined_search(self, query_vector):
        """Perform a refined search using updated feature weights."""
        print("Performing refined search.")
        weighted_query = query_vector * self.weights
        similarities = []
        for image_id in self.image_ids:
            descriptor = np.array(self.database[image_id]) * self.weights
            similarity = cosine_similarity(weighted_query.reshape(1, -1), descriptor.reshape(1, -1))[0, 0]
            similarities.append((image_id, similarity))
            sorted_images = sorted(similarities, key=lambda x: x[1], reverse=True)
        return sorted_images[:20]
    def reset_feedback(self):
        """Reset feedback to allow initial search for new queries."""
        self.weights = np.ones(len(next(iter(self.database.values()))))  # Reset weights
        self.feedback_received = False