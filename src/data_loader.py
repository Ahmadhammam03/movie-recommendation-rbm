"""
Data loading and preprocessing utilities for RBM
Author: Ahmad Hammam
"""

import numpy as np
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split


class MovieLensDataLoader:
    """
    Data loader for MovieLens datasets with RBM-specific preprocessing.
    """
    
    def __init__(self, data_path='data/', dataset='ml-100k'):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to data directory
            dataset: Which dataset to use ('ml-100k' or 'ml-1m')
        """
        self.data_path = data_path
        self.dataset = dataset
        self.movies = None
        self.users = None
        self.ratings = None
        
    def load_metadata(self):
        """Load movie and user metadata."""
        print(f"Loading {self.dataset} metadata...")
        
        if self.dataset == 'ml-1m':
            # Load ML-1M metadata
            movies_path = os.path.join(self.data_path, 'ml-1m', 'movies.dat')
            users_path = os.path.join(self.data_path, 'ml-1m', 'users.dat')
            ratings_path = os.path.join(self.data_path, 'ml-1m', 'ratings.dat')
            
            self.movies = pd.read_csv(movies_path, sep='::', header=None,
                                    names=['MovieID', 'Title', 'Genres'],
                                    engine='python', encoding='latin-1')
            
            self.users = pd.read_csv(users_path, sep='::', header=None,
                                   names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip'],
                                   engine='python', encoding='latin-1')
            
            self.ratings = pd.read_csv(ratings_path, sep='::', header=None,
                                     names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                                     engine='python', encoding='latin-1')
        else:
            # For ML-100K, load what's available
            self.movies = None  # ML-100K has different format
            self.users = None
            self.ratings = None
            
        print(f"Metadata loaded successfully!")
        
    def load_train_test_split(self):
        """Load predefined train/test split."""
        if self.dataset == 'ml-100k':
            # Load u1.base and u1.test
            train_path = os.path.join(self.data_path, 'ml-100k', 'u1.base')
            test_path = os.path.join(self.data_path, 'ml-100k', 'u1.test')
            
            training_set = pd.read_csv(train_path, delimiter='\t', header=None,
                                     names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
            test_set = pd.read_csv(test_path, delimiter='\t', header=None,
                                 names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
            
            # Convert to numpy arrays
            training_set = np.array(training_set[['UserID', 'MovieID', 'Rating']], dtype='int')
            test_set = np.array(test_set[['UserID', 'MovieID', 'Rating']], dtype='int')
            
        else:
            # For ML-1M, create split from ratings
            if self.ratings is None:
                self.load_metadata()
                
            # Create 80-20 split
            ratings_array = np.array(self.ratings[['UserID', 'MovieID', 'Rating']], dtype='int')
            training_set, test_set = train_test_split(ratings_array, test_size=0.2, random_state=42)
            
        return training_set, test_set
    
    def convert_to_matrix(self, data, nb_users, nb_movies):
        """
        Convert rating data to user-movie matrix.
        
        Args:
            data: Rating data array
            nb_users: Total number of users
            nb_movies: Total number of movies
            
        Returns:
            List of user rating vectors
        """
        matrix = []
        
        for user_id in range(1, nb_users + 1):
            user_ratings = data[data[:, 0] == user_id]
            
            ratings_vector = np.zeros(nb_movies)
            if len(user_ratings) > 0:
                movie_indices = user_ratings[:, 1] - 1  # 0-based indexing
                ratings_vector[movie_indices] = user_ratings[:, 2]
            
            matrix.append(list(ratings_vector))
            
        return matrix
    
    def prepare_rbm_data(self, binary=True):
        """
        Prepare data for RBM training.
        
        Args:
            binary: Whether to convert to binary ratings
            
        Returns:
            Tuple of (training_set, test_set, nb_users, nb_movies)
        """
        # Load train/test split
        train_array, test_array = self.load_train_test_split()
        
        # Get dimensions
        nb_users = int(max(max(train_array[:, 0]), max(test_array[:, 0])))
        nb_movies = int(max(max(train_array[:, 1]), max(test_array[:, 1])))
        
        print(f"Number of users: {nb_users}")
        print(f"Number of movies: {nb_movies}")
        
        # Convert to matrices
        training_set = self.convert_to_matrix(train_array, nb_users, nb_movies)
        test_set = self.convert_to_matrix(test_array, nb_users, nb_movies)
        
        # Convert to torch tensors
        training_set = torch.FloatTensor(training_set)
        test_set = torch.FloatTensor(test_set)
        
        if binary:
            # Convert to binary ratings
            print("Converting to binary ratings...")
            training_set = self.binarize_ratings(training_set)
            test_set = self.binarize_ratings(test_set)
            
        return training_set, test_set, nb_users, nb_movies
    
    def binarize_ratings(self, ratings):
        """
        Convert ratings to binary format for RBM.
        
        0 → -1 (no rating)
        1-2 → 0 (not liked)
        3-5 → 1 (liked)
        """
        ratings[ratings == 0] = -1
        ratings[ratings == 1] = 0
        ratings[ratings == 2] = 0
        ratings[ratings >= 3] = 1
        
        return ratings
    
    def get_movie_info(self, movie_id):
        """Get movie title and genres."""
        if self.movies is not None:
            movie = self.movies[self.movies['MovieID'] == movie_id]
            if len(movie) > 0:
                return movie.iloc[0]['Title'], movie.iloc[0]['Genres']
        return f"Movie {movie_id}", "Unknown"
    
    def get_user_info(self, user_id):
        """Get user demographics."""
        if self.users is not None:
            user = self.users[self.users['UserID'] == user_id]
            if len(user) > 0:
                return user.iloc[0].to_dict()
        return {"UserID": user_id}
    
    def get_data_statistics(self, training_set, test_set):
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with various statistics
        """
        # Count non-zero ratings
        train_ratings = (training_set >= 0).sum().item()
        test_ratings = (test_set >= 0).sum().item()
        
        # Sparsity
        total_possible = training_set.shape[0] * training_set.shape[1]
        sparsity = 1 - (train_ratings / total_possible)
        
        # Rating distribution (for non-binary data)
        stats = {
            'n_users': training_set.shape[0],
            'n_movies': training_set.shape[1],
            'n_train_ratings': train_ratings,
            'n_test_ratings': test_ratings,
            'sparsity': sparsity,
            'avg_ratings_per_user': train_ratings / training_set.shape[0],
            'avg_ratings_per_movie': train_ratings / training_set.shape[1]
        }
        
        return stats


class DataPreprocessor:
    """
    Additional preprocessing utilities for RBM.
    """
    
    @staticmethod
    def normalize_ratings(ratings, method='minmax'):
        """
        Normalize ratings to [0, 1] range.
        
        Args:
            ratings: Rating matrix
            method: Normalization method ('minmax' or 'zscore')
        """
        mask = ratings >= 0  # Only normalize actual ratings
        
        if method == 'minmax':
            # Min-max normalization
            min_val = 1.0
            max_val = 5.0
            ratings[mask] = (ratings[mask] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            # Z-score normalization
            mean = ratings[mask].mean()
            std = ratings[mask].std()
            ratings[mask] = (ratings[mask] - mean) / std
            
        return ratings
    
    @staticmethod
    def add_noise(data, noise_level=0.1):
        """
        Add noise to training data for denoising autoencoder variant.
        
        Args:
            data: Input data
            noise_level: Fraction of values to corrupt
        """
        noisy_data = data.clone()
        mask = data >= 0  # Only add noise to actual ratings
        
        # Random corruption
        corruption = torch.rand_like(data) < noise_level
        noisy_data[mask & corruption] = 1 - noisy_data[mask & corruption]
        
        return noisy_data
    
    @staticmethod
    def create_minibatches(data, batch_size, shuffle=True):
        """
        Create minibatches for training.
        
        Args:
            data: Training data
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of data
        """
        n_samples = len(data)
        indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield data[batch_indices]


# Example usage
if __name__ == "__main__":
    # Test data loader
    loader = MovieLensDataLoader(data_path='../data/')
    
    # Load and prepare data
    train_set, test_set, n_users, n_movies = loader.prepare_rbm_data(binary=True)
    
    # Get statistics
    stats = loader.get_data_statistics(train_set, test_set)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")