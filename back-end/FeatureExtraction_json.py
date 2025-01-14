import os
import cv2
import numpy as np
import json
from skimage.feature import local_binary_pattern, hog
from sklearn.cluster import KMeans

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


# Function to calculate all descriptors for an image
def calculate_image_descriptors(image_path):
    hist_r, hist_g, hist_b = calculate_normalized_histogram(image_path)
    contrast, dissimilarity, homogeneity, energy = calculate_co_occurrence_matrix(image_path)
    hog_features = calculate_hog(image_path)
    lbp_features = calculate_lbp(image_path)
    hu_moments = calculate_hu_moments(image_path)
    gabor_features = calculate_gabor_features(image_path)
    dominant_colors = calculate_dominant_colors(image_path)

    # Concatenate all descriptors
    descriptor = np.concatenate((
        hist_r,
        hist_g,
        hist_b,
        np.array([contrast, dissimilarity, homogeneity, energy]),
        hog_features,
        lbp_features,
        hu_moments,
        gabor_features,
        dominant_colors
    ))

    return descriptor


# Fonction pour créer la base de données des descripteurs
def create_descriptor_database(directory_path):
    database = {}

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(root, file)

                try:
                    descriptor = calculate_image_descriptors(image_path)
                    database[image_path] = descriptor.tolist()
                    print(f"Descripteurs calculés pour {image_path}")
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {image_path}: {e}")

    return database

# Main execution starts here
directory_path = input("Entrez le chemin du dossier contenant les images : ")

# Créer la base de données de descripteurs
database = create_descriptor_database(directory_path)

# Sauvegarder la base de données dans un fichier JSON
with open('database.json', 'w') as f:
    json.dump(database, f)