import os
import pandas as pd
from tqdm import tqdm
import trimesh
import numpy as np
import pymeshlab  # Using PyMeshLab for mesh reduction
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from scipy.special import sph_harm
from typing import List, Tuple

import matplotlib.pyplot as plt

def reduce_mesh_with_pymeshlab(mesh: trimesh.Trimesh, reduction_factor: float) -> trimesh.Trimesh:
    """
    Reduce the complexity of a 3D mesh using PyMeshLab.

    Args:
        mesh (trimesh.Trimesh): The original mesh.
        reduction_factor (float): The proportion of the original vertices to retain (0.0 - 1.0).

    Returns:
        trimesh.Trimesh: The reduced mesh.
    """
    if reduction_factor <= 0 or reduction_factor > 1:
        raise ValueError("Reduction factor must be between 0 and 1 (exclusive).")

    try:

        # Initialize PyMeshLab MeshSet
        ms = pymeshlab.MeshSet()
        

        # Add the original mesh to MeshSet
        ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))

        # Target number of faces for simplification
        target_faces = int(len(mesh.faces) * reduction_factor)

        # Use quadric edge collapse decimation
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)

        # Retrieve the simplified mesh
        reduced = ms.current_mesh()
        reduced_mesh = trimesh.Trimesh(vertices=reduced.vertex_matrix(), faces=reduced.face_matrix())
        return reduced_mesh
    except Exception as e:
        print(f"Error during mesh reduction: {e}")
        return mesh  # Return the original mesh if reduction fails


def compute_fourier_features(mesh: trimesh.Trimesh) -> float:
    """
    Compute Fourier features from a 3D mesh.

    Returns:
        float: Simplified Fourier feature as mean FFT magnitude.
    """
    mesh.apply_translation(-mesh.center_mass)
    mesh.apply_scale(1 / mesh.scale)
    fourier_coefficients = np.fft.fftn(mesh.vertices)
    return np.mean(np.abs(fourier_coefficients))


def compute_zernike_features(mesh: trimesh.Trimesh) -> float:
    """
    Compute a simplified Zernike-like feature using mesh vertex statistics.

    Returns:
        float: Simplified Zernike feature.
    """
    vertices = mesh.vertices - np.mean(mesh.vertices, axis=0)
    vertices /= np.max(np.linalg.norm(vertices, axis=1))
    return np.mean(np.std(vertices, axis=0))



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

