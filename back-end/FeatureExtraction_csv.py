import os
import cv2
import numpy as np
import csv
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
# Function to calculate all descriptors and save results to CSV
def process_images_and_save_to_csv(directory_path, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Average Histogram", "Average Co-occurrence", "Average HOG", "Average LBP", "Average Hu Moments", "Average Gabor Features", "Average Dominant Colors"])

        for root, _, files in os.walk(directory_path):
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(root, file_name)
                    category = os.path.basename(root)  # Get the category name (folder name)
                    image_category_name = f"{category}/{file_name}"  # Format category/filename

                    try:
                        avg_hist = np.mean(calculate_normalized_histogram(image_path))
                        avg_co_occurrence = np.mean(calculate_co_occurrence_matrix(image_path))  # Ensure it computes the mean
                        avg_hog = np.mean(calculate_hog(image_path))
                        avg_lbp = np.mean(calculate_lbp(image_path))
                        avg_hu = np.mean(calculate_hu_moments(image_path))
                        avg_gabor = np.mean(calculate_gabor_features(image_path))
                        avg_dominant_colors = np.mean(calculate_dominant_colors(image_path))

                        writer.writerow([
                            image_category_name,
                            avg_hist,
                            avg_co_occurrence,
                            avg_hog,
                            avg_lbp,
                            avg_hu,
                            avg_gabor,
                            avg_dominant_colors
                        ])

                        print(f"Processed: {image_category_name}")

                    except Exception as e:
                        print(f"Error processing {image_category_name}: {e}")


# Main execution
directory_path = input("Enter the path of the directory containing images: ")
output_csv = "image_descriptors.csv"
process_images_and_save_to_csv(directory_path, output_csv)
print(f"Results saved to {output_csv}")
