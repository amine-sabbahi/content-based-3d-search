# simple_search2.py
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
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

    avg_r = np.mean(hist_r)
    avg_g = np.mean(hist_g)
    avg_b = np.mean(hist_b)

    return avg_r, avg_g, avg_b

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

    avg_co_occurrence = np.mean([contrast, dissimilarity, homogeneity, energy])

    return avg_co_occurrence

# Fonction pour calculer les descripteurs HOG (Histogram of Oriented Gradients)
def calculate_hog(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossible de lire l'image à {image_path}. Vérifiez le chemin.")

    fd, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    avg_hog = np.mean(fd)

    return avg_hog

# Fonction pour calculer le LBP (Local Binary Pattern)
def calculate_lbp(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossible de lire l'image à {image_path}. Vérifiez le chemin.")

    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')

    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= lbp_hist.sum()

    avg_lbp = np.mean(lbp_hist)

    return avg_lbp

# Fonction pour calculer les moments de Hu
def calculate_hu_moments(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossible de lire l'image à {image_path}. Vérifiez le chemin.")

    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()

    avg_hu = np.mean(hu_moments)

    return avg_hu

# Function to calculate Gabor filter features
def calculate_gabor_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image at {image_path}. Check the path.")

    kernels = []
    gabor_features = []

    for theta in range(4):
        theta = theta * np.pi / 4
        kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)

    for kernel in kernels:
        filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        gabor_features.append(np.mean(filtered_image))
        gabor_features.append(np.std(filtered_image))

    avg_gabor = np.mean(gabor_features)

    return avg_gabor

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

    avg_dominant_colors = np.mean(dominant_colors)

    return avg_dominant_colors

# Function to calculate all descriptors for an image
def calculate_image_descriptors(image_path):
    """
    Calculate multiple image descriptors and combine them into a single feature vector.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Combined feature vector containing all descriptors
        
    Raises:
        ValueError: If image cannot be read or processed
    """
    # Input validation
    if not os.path.exists(image_path):
        raise ValueError(f"Image path does not exist: {image_path}")
        
    try:
        # Calculate individual descriptors
        hist_r, hist_g, hist_b = calculate_normalized_histogram(image_path)
        avg_co_occurrence = calculate_co_occurrence_matrix(image_path)
        hog_features = calculate_hog(image_path)
        lbp_features = calculate_lbp(image_path)
        hu_moments = calculate_hu_moments(image_path)
        gabor_features = calculate_gabor_features(image_path)
        dominant_colors = calculate_dominant_colors(image_path)

        # Create feature vector ensuring all elements are numpy arrays
        descriptor_list = [
            np.array([hist_r, hist_g, hist_b]),
            np.array([avg_co_occurrence]),
            np.array([hog_features]),
            np.array([lbp_features]),
            np.array([hu_moments]),
            np.array([gabor_features]),
            np.array([dominant_colors])
        ]

        # Verify each descriptor's shape and type
        for i, desc in enumerate(descriptor_list):
            if not isinstance(desc, np.ndarray):
                descriptor_list[i] = np.array(desc)
            if desc.size == 0:
                raise ValueError(f"Empty descriptor detected at index {i}")

        # Combine all descriptors into a single feature vector
        combined_descriptor = np.concatenate([d.flatten() for d in descriptor_list])
        
        # Normalize the combined descriptor
        if np.any(np.isnan(combined_descriptor)) or np.any(np.isinf(combined_descriptor)):
            raise ValueError("Invalid values (NaN or Inf) detected in descriptors")
            
        return combined_descriptor

    except Exception as e:
        raise ValueError(f"Error calculating descriptors: {str(e)}")

# Function to compare a new image with the database
def compare_images(new_image_path, database_df, descriptor_type):
    """
    Compare a new image with the database using cosine similarity.
    
    Args:
        new_image_path (str): Path to the new image
        database_df (pd.DataFrame): Database containing image descriptors
        descriptor_type (str): Type of descriptor to use for comparison
        
    Returns:
        list: Top 20 similar images with their cosine distances
    """
    from scipy.spatial.distance import cosine
    
    # Calculate descriptors for the new image
    hist_r, hist_g, hist_b = calculate_normalized_histogram(new_image_path)
    avg_hist = np.mean([hist_r, hist_g, hist_b])
    
    # Get all descriptors for the new image
    new_descriptors = {
        'histogram_color': avg_hist,
        'co-occurrence': calculate_co_occurrence_matrix(new_image_path),
        'hog': calculate_hog(new_image_path),
        'lbp': calculate_lbp(new_image_path),
        'hu': calculate_hu_moments(new_image_path),
        'gabor': calculate_gabor_features(new_image_path),
        'dominant_colors': calculate_dominant_colors(new_image_path)
    }
    
    # Column mapping in the database
    column_mapping = {
        'histogram_color': 'Average Histogram',
        'co-occurrence': 'Average Co-occurrence',
        'hog': 'Average HOG',
        'lbp': 'Average LBP',
        'hu': 'Average Hu Moments',
        'gabor': 'Average Gabor Features',
        'dominant_colors': 'Average Dominant Colors'
    }
    
    distances = {}
    
    if descriptor_type == 'all':
        # For all descriptors, create feature vectors for cosine similarity
        for _, row in database_df.iterrows():
            # Create feature vectors for both new image and database image
            new_vector = np.array([new_descriptors[desc_type] for desc_type in column_mapping.keys()])
            db_vector = np.array([row[column_name] for column_name in column_mapping.values()])
            
            # Handle potential zero vectors
            if np.all(new_vector == 0) or np.all(db_vector == 0):
                distances[row['Image Name']] = 1.0  # Maximum distance for zero vectors
            else:
                # Calculate cosine distance
                distances[row['Image Name']] = cosine(new_vector, db_vector)
    else:
        # For single descriptor
        if descriptor_type not in column_mapping:
            raise ValueError(f"Invalid descriptor type. Choose from: {', '.join(column_mapping.keys())}")
            
        column_name = column_mapping[descriptor_type]
        new_value = np.array([new_descriptors[descriptor_type]])
        
        for _, row in database_df.iterrows():
            db_value = np.array([row[column_name]])
            
            # Handle potential zero vectors
            if np.all(new_value == 0) or np.all(db_value == 0):
                distances[row['Image Name']] = 1.0
            else:
                distances[row['Image Name']] = cosine(new_value, db_value)
    
    # Sort by distance (lower cosine distance means more similar)
    sorted_images = sorted(distances.items(), key=lambda x: x[1])
    return sorted_images[:20]
