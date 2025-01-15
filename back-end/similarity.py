import os
import pandas as pd
import trimesh
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from scipy.special import sph_harm
from typing import List, Tuple

import matplotlib.pyplot as plt


def compute_fourier_features(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Compute Fourier features from a 3D mesh by performing FFT on vertex coordinates.
    """
    # Normalize mesh position and scale
    mesh.apply_translation(-mesh.center_mass)
    mesh.apply_scale(1 / mesh.scale)
    
    # Compute FFT and get magnitude (rotation-invariant)
    fourier_coefficients = np.fft.fftn(mesh.vertices)
    return np.abs(fourier_coefficients)

def compute_factorial(n: int) -> int:
    """Compute factorial using recursion."""
    return 1 if n == 0 else n * compute_factorial(n - 1)

def compute_zernike_basis(n: int, l: int, m: int, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Compute 3D Zernike basis function."""
    # Compute radial polynomial
    radial_poly = 0
    for k in range((n - l) // 2 + 1):
        numerator = ((-1) ** k) * compute_factorial(n - k)
        denominator = (
            compute_factorial(k) * 
            compute_factorial((n + l) // 2 - k) * 
            compute_factorial((n - l) // 2 - k)
        )
        radial_poly += (numerator / denominator) * (r ** (n - 2 * k))
    
    # Combine with spherical harmonics
    return radial_poly * sph_harm(m, l, phi, theta)

def compute_zernike_moments(mesh: trimesh.Trimesh, max_order: int = 10) -> dict:
    """Compute 3D Zernike moments for a mesh."""
    # Center and normalize vertices
    vertices = mesh.vertices
    vertices = vertices - np.mean(vertices, axis=0)
    vertices = vertices / np.max(np.linalg.norm(vertices, axis=1))
    
    # Convert to spherical coordinates
    x, y, z = vertices.T
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / np.where(r != 0, r, 1))
    phi = np.arctan2(y, x)
    
    # Compute moments
    moments = {}
    for n in range(max_order + 1):
        for l in range(n + 1):
            if (n - l) % 2 != 0:
                continue
            for m in range(-l, l + 1):
                basis_values = compute_zernike_basis(n, l, m, r, theta, phi)
                weight = 1.0 / len(vertices)
                moments[(n, l, m)] = np.abs(np.sum(basis_values * weight))
    
    return moments

def extract_shape_features(mesh_path: str, max_order: int = 10) -> np.ndarray:
    """Extract combined shape features."""
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Compute and normalize Fourier features
    fourier_features = compute_fourier_features(mesh)
    fourier_features = fourier_features.flatten()
    fourier_features = fourier_features / np.linalg.norm(fourier_features)
    
    # Compute and normalize Zernike features
    zernike_features = compute_zernike_moments(mesh, max_order)
    zernike_features = np.array(list(zernike_features.values()))
    zernike_features = zernike_features / np.linalg.norm(zernike_features)
    
    # Return combined features
    return np.array([
        np.mean(fourier_features),
        np.mean(zernike_features)
    ])


def load_database(csv_path: str) -> pd.DataFrame:
    """
    Load the 3D model database from a CSV file.
    """
    return pd.read_csv(csv_path)

def find_most_similar(query_features, database, k=10):
    query_fourier, query_zernike = query_features
    similarities = []

    # Iterate through the database rows
    for _, row in database.iterrows():
        
        # Calculate the similarity
        similarity = (
    euclidean(np.array([query_fourier]), np.array([row["fourier_feature"]])) +
    euclidean(np.array([query_zernike]), np.array([row["zernike_feature"]]))
)
        
        # Append to the list along with relevant details
        similarities.append({
            "model_path": row["model_path"],  # Adjust column name as per your database
            "thumbnail": row["thumbnail_path"],    # Adjust column name for thumbnails
            "category": row["category"],      # Adjust column name for category
            "similarity": similarity
        })
    
    # Sort the list by similarity
    sorted_similarities = sorted(similarities, key=lambda x: x["similarity"])
    
    # Get top k results
    top_results = sorted_similarities[:k]
    print(top_results)
    


    return top_results

def display_results(results):
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    
    for i, result in enumerate(results):
        ax = axes[i]
        thumbnail = plt.imread(result["thumbnail"])  # Load thumbnail image
        ax.imshow(thumbnail)
        ax.set_title(f"Category: {result['category']}\nSim: {result['similarity']:.6f}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


