import h5py
import dask.array as da
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from cuml.dask.decomposition import PCA as daskPCA
import numpy as np
import sys
import time
from contextlib import contextmanager

@contextmanager
def dask_cluster(n_workers=1, device_memory_limit='70%'):
    """Context manager for handling Dask cluster lifecycle with Colab support"""
    cluster = None
    client = None
    try:
        # Shutdown any existing clusters
        try:
            client = Client()
            client.shutdown()
            client.close()
            print("Existing Dask cluster shut down.")
        except:
            pass

        # For Google Colab, set up port forwarding
        from google.colab import output
        output.serve_kernel_port_as_window(8787)

        # Create new cluster
        cluster = LocalCUDACluster(
            n_workers=n_workers,
            device_memory_limit=device_memory_limit,
            dashboard_address=':8787',  # Use port 8787 for Colab
            silence_logs=False
        )
        client = Client(cluster)

        # Print the Colab-specific dashboard link
        print("The dashboard will open in a new browser tab.")
        print("If it doesn't open automatically, check the 'View' -> 'Output' -> 'Port 8787' menu in Colab")

        yield client

    finally:
        if client:
            client.close()
        if cluster:
            cluster.close()

def run_pca_pipeline(
    input_h5_filename,
    output_h5_filename,
    n_components=10,
    chunk_size=1000,
    device_memory_limit=0.7
):
    """Main PCA pipeline with improved error handling and memory management"""

    with dask_cluster(device_memory_limit=device_memory_limit) as client:
        try:
            with h5py.File(input_h5_filename, 'r') as h5f:
                data = h5f['data']
                total_samples, width, height = data.shape
                print(f"Original data shape: {data.shape}")

                # Calculate the total number of features by multiplying width and height
                total_features = width * height
                print(f"Total features after flattening: {total_features}")

                # Create a Dask array from the HDF5 dataset with appropriate chunking
                # Initial chunking: (chunk_size, width, height)
                dask_array = da.from_array(data, chunks=(chunk_size, width, height))
                print(f"Initial Dask array shape: {dask_array.shape}")
                print(f"Initial Dask array chunks: {dask_array.chunks}")

                # Reshape to 2D: (n_samples, n_features)
                # After reshaping, chunks should be (chunk_size, n_features)
                dask_array_2d = dask_array.reshape((total_samples, total_features))
                print(f"Reshaped Dask array shape: {dask_array_2d.shape}")
                print(f"Reshaped Dask array chunks: {dask_array_2d.chunks}")

                # Verify that the reshaped array is 2D
                if dask_array_2d.ndim != 2:
                    raise ValueError(f"Reshaped array has {dask_array_2d.ndim} dimensions; expected 2D array.")

                # Initialize PCA
                pca = daskPCA(n_components=n_components)

                print("Fitting PCA on the entire dataset...")
                pca.fit(dask_array_2d)

                print("Transforming the entire dataset...")
                X_pca = pca.transform(dask_array_2d).compute()

                # Save results
                with h5py.File(output_h5_filename, 'w') as h5f_out:
                    h5f_out.create_dataset('pca_data', data=X_pca)
                    h5f_out.create_dataset('explained_variance_ratio', data=pca.explained_variance_ratio_)
                    h5f_out.create_dataset('components', data=pca.components_)

                print(f"PCA transformation completed successfully. Results saved to {output_h5_filename}")

        except Exception as e:
            print(f"Error during PCA processing: {str(e)}")
            raise
