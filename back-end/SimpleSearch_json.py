# SimpleSearch.py
import os
import cv2
import numpy as np
import json
from scipy.spatial.distance import cosine
from skimage.feature import local_binary_pattern, hog
from sklearn.cluster import KMeans
# Function to calculate normalized color histogram
def calculate_normalized_histogram(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read the image at {image_path}. Check the path.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    histSize = 256
    hist_r = cv2.calcHist([image], [0], None, [histSize], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [histSize], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [histSize], [0, 256])

    # Normalize histograms
    cv2.normalize(hist_r, hist_r, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_g, hist_g, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist_r.flatten(), hist_g.flatten(), hist_b.flatten()

# Function to calculate co-occurrence matrix
def calculate_co_occurrence_matrix(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read the image at {image_path}. Check the path.")

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

# Function to calculate HOG descriptors
def calculate_hog(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read the image at {image_path}. Check the path.")

    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return fd

# Function to calculate LBP
def calculate_lbp(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read the image at {image_path}. Check the path.")

    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')

    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= lbp_hist.sum()

    return lbp_hist

# Function to calculate Hu moments
def calculate_hu_moments(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read the image at {image_path}. Check the path.")

    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
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




def compare_images(new_image_path, database, descriptor_type='all'):
    """
    Compare a new image with the database using specified descriptor type
    
    Args:
    - new_image_path (str): Full path to the new image
    - database (dict): Dictionary of image descriptors 
    - descriptor_type (str): Type of descriptor to use for comparison
    
    Returns:
    - List of tuples with (image_path, distance) sorted by similarity
    """
    try:
        # Calculate descriptors for the new image
        new_image_descriptor = calculate_image_descriptors(new_image_path)
        
        
        new_image_descriptor = new_image_descriptor.tolist()
        

                # Slice the descriptor based on the selected type
        if descriptor_type == 'histogram_color':
            new_image_descriptor = new_image_descriptor[:768]
    
        elif descriptor_type == 'co-occurrence': 
            new_image_descriptor = new_image_descriptor[768:772]

        elif descriptor_type == 'hog': 
            new_image_descriptor = new_image_descriptor[772:87208]

        elif descriptor_type == 'lbp': 
            new_image_descriptor = new_image_descriptor[87208:87218]
 
        elif descriptor_type == 'hu':
            new_image_descriptor = new_image_descriptor[87218:87225]

        elif descriptor_type == 'gabor':
            new_image_descriptor = new_image_descriptor[87225:87233]
  
        elif descriptor_type == 'dominant_colors':
            new_image_descriptor = new_image_descriptor[87233:]
 
        elif descriptor_type == 'all':
            new_image_descriptor = new_image_descriptor
            
        
       
        distances = {}
        for image_path, descriptor in database.items():
        
         if descriptor_type == 'histogram_color':
            # Return the histogram of color as 3 arrays (R, G, B)
            descriptor = descriptor[:768]  # First 768 for red green blue channels HISTO
           
           
         elif descriptor_type == 'co-occurence': 
            descriptor = descriptor[768:772]  # 4 for coocurence
            
            
         elif descriptor_type == 'hog': 
            descriptor = descriptor[772:87208]  # 86436 for hog
            
            
         elif descriptor_type == 'lbp': 
            descriptor = descriptor[87208:87218]  # 10 for lbp
           
            
         elif descriptor_type == 'hu':
            descriptor = descriptor[87218:87225]  # 7 for hu moments
            
            
         elif descriptor_type == 'gabor':
            descriptor = descriptor[87225:87233]  # 8 for gabor
            
            
         elif descriptor_type == 'dominant_colors':
            descriptor = descriptor[87233:]  # 15 for dominant colors
            
            
         elif descriptor_type == 'all':
            descriptor = descriptor  # all
            
            
         distance = cosine(new_image_descriptor, descriptor)
         distances[image_path] = distance
        
        
        
        
        # Sort and return top 20 most similar images
        sorted_images = sorted(distances.items(), key=lambda x: x[1])
        return sorted_images[:20]
    
    except Exception as e:
        print(f"Error in compare_images: {e}")
        import traceback
        traceback.print_exc()
        return []
