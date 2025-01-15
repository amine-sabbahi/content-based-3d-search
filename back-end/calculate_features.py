import os
import pandas as pd
from tqdm import tqdm
import trimesh
import numpy as np
from typing import List, Tuple
from scipy.special import sph_harm

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

def find_matching_files(models_dir: str, thumbnails_dir: str) -> List[Tuple[str, str, str]]:
    """Find matching 3D models and thumbnails."""
    matching_files = []
    
    for category in os.listdir(models_dir):
        category_models_path = os.path.join(models_dir, category)
        category_thumbnails_path = os.path.join(thumbnails_dir, category)
        
        if not os.path.isdir(category_models_path):
            continue
            
        for model_file in os.listdir(category_models_path):
            if not model_file.endswith('.obj'):
                continue
                
            base_name = os.path.splitext(model_file)[0]
            thumbnail_file = f"{base_name}.jpg"
            thumbnail_path = os.path.join(category_thumbnails_path, thumbnail_file)
            model_path = os.path.join(category_models_path, model_file)
            
            if os.path.exists(thumbnail_path):
                matching_files.append((model_path, thumbnail_path, category))
    
    return matching_files

def process_dataset(dataset_root: str) -> pd.DataFrame:
    """Process the entire dataset and create a DataFrame with results."""
    # Setup paths
    models_dir = os.path.join(dataset_root, "3D Models")
    thumbnails_dir = os.path.join(dataset_root, "Thumbnails")
    
    # Find all matching files
    print("Finding matching files...")
    matching_files = find_matching_files(models_dir, thumbnails_dir)
    
    # Prepare data storage
    data = []
    
    # Process each file
    print("Processing 3D models...")
    for model_path, thumbnail_path, category in tqdm(matching_files):
        try:
            # Extract features
            features = extract_shape_features(
                mesh_path=model_path,
                max_order=10
            )
            
            # Store results
            data.append({
                'model_path': model_path,
                'thumbnail_path': thumbnail_path,
                'category': category,
                'fourier_feature': features[0],
                'zernike_feature': features[1]
            })
            
        except Exception as e:
            print(f"\nError processing {model_path}: {str(e)}")
            # Add entry with error flag
            data.append({
                'model_path': model_path,
                'thumbnail_path': thumbnail_path,
                'category': category,
                'fourier_feature': np.nan,
                'zernike_feature': np.nan,
                'error': str(e)
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def main():
    # Set the path to your dataset root directory
    dataset_root = "3d-dataset"
    
    # Process the dataset
    print("Starting dataset processing...")
    results_df = process_dataset(dataset_root)
    
    # Save results
    output_path = "dataset_features2.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Total processed objects: {len(results_df)}")
    print("\nSuccessfully processed objects: {len(results_df.dropna())}")
    print("\nCategory distribution:")
    print(results_df['category'].value_counts())
    
    # Print error summary if any
    failed_cases = results_df[results_df['fourier_feature'].isna()]
    if len(failed_cases) > 0:
        print(f"\nFailed cases: {len(failed_cases)}")
        print("\nError distribution:")
        if 'error' in failed_cases.columns:
            print(failed_cases['error'].value_counts())

if __name__ == "__main__":
    main()