import os
import pandas as pd
from tqdm import tqdm
import trimesh
import numpy as np
import pymeshlab  # Using PyMeshLab for mesh reduction


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
        ms.simplification_quadric_edge_collapse_decimation(targetfacenum=target_faces)

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


def process_reduced_dataset(input_csv: str, output_csv: str, reduction_factor: float) -> None:
    """
    Process the dataset with mesh reduction and recompute features.

    Args:
        input_csv (str): Path to the original dataset file.
        output_csv (str): Path to save the reduced dataset.
        reduction_factor (float): Reduction factor to apply to meshes.
    """
    # Load the existing dataset
    dataset = pd.read_csv(input_csv)
    data = []

    print(f"Processing dataset with mesh reduction (factor: {reduction_factor})...")
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        model_path = row['model_path']
        thumbnail_path = row['thumbnail_path']
        category = row['category']

        try:
            # Load and reduce the mesh
            original_mesh = trimesh.load(model_path)
            reduced_mesh = reduce_mesh_with_pymeshlab(original_mesh, reduction_factor)

            # Compute features
            fourier_feature = compute_fourier_features(reduced_mesh)
            zernike_feature = compute_zernike_features(reduced_mesh)

            # Store results
            data.append({
                'model_path': model_path,
                'thumbnail_path': thumbnail_path,
                'category': category,
                'fourier_feature': fourier_feature,
                'zernike_feature': zernike_feature
            })

        except Exception as e:
            print(f"Error processing {model_path}: {e}")
            data.append({
                'model_path': model_path,
                'thumbnail_path': thumbnail_path,
                'category': category,
                'fourier_feature': np.nan,
                'zernike_feature': np.nan,
                'error': str(e)
            })

    # Save the new dataset
    reduced_df = pd.DataFrame(data)
    reduced_df.to_csv(output_csv, index=False)
    print(f"Reduced dataset saved to {output_csv}")


def main():
    input_csv = "dataset_features.csv"  # Path to the original dataset

    # Generate datasets with 20%, 50%, and 70% reduction
    reduction_factors = {
        "dataset_features_reduced_201.csv": 0.2,
        "dataset_features_reduced_501.csv": 0.5,
        "dataset_features_reduced_701.csv": 0.7
    }

    for output_csv, reduction_factor in reduction_factors.items():
        process_reduced_dataset(input_csv, output_csv, reduction_factor)


if __name__ == "__main__":
    main()
