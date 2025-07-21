"""
Main training script for Restricted Boltzmann Machine (RBM) movie recommendation system.
Author: Ahmad Hammam
GitHub: https://github.com/Ahmadhammam03
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os
import time
from tqdm import tqdm


class RBM:
    """
    Restricted Boltzmann Machine for collaborative filtering.
    
    Uses energy-based learning with Gibbs sampling for recommendations.
    """
    
    def __init__(self, nv, nh):
        """
        Initialize RBM.
        
        Args:
            nv (int): Number of visible units (movies)
            nh (int): Number of hidden units
        """
        self.W = torch.randn(nh, nv)  # Weight matrix
        self.a = torch.randn(1, nh)   # Hidden bias
        self.b = torch.randn(1, nv)   # Visible bias
        
    def sample_h(self, x):
        """
        Sample hidden units given visible units.
        
        Args:
            x: Visible layer state
            
        Returns:
            Tuple of (probability, sample)
        """
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        """
        Sample visible units given hidden units.
        
        Args:
            y: Hidden layer state
            
        Returns:
            Tuple of (probability, sample)
        """
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        """
        Update weights using Contrastive Divergence.
        
        Args:
            v0: Initial visible state
            vk: Visible state after k Gibbs steps
            ph0: Initial hidden probabilities
            phk: Hidden probabilities after k steps
        """
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)


class MovieRecommenderRBM:
    """Complete movie recommendation system using RBM."""
    
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.rbm = None
        self.nb_users = None
        self.nb_movies = None
        
    def load_data(self):
        """Load MovieLens dataset."""
        print("Loading MovieLens datasets...")
        
        # Load metadata
        ml1m_path = os.path.join(self.data_path, 'ml-1m')
        ml100k_path = os.path.join(self.data_path, 'ml-100k')
        
        # Load movie and user information
        self.movies = pd.read_csv(os.path.join(ml1m_path, 'movies.dat'), 
                                sep='::', header=None, engine='python', 
                                encoding='latin-1')
        self.users = pd.read_csv(os.path.join(ml1m_path, 'users.dat'), 
                               sep='::', header=None, engine='python', 
                               encoding='latin-1')
        self.ratings = pd.read_csv(os.path.join(ml1m_path, 'ratings.dat'), 
                                 sep='::', header=None, engine='python', 
                                 encoding='latin-1')
        
        # Load training and test sets
        self.training_set = pd.read_csv(os.path.join(ml100k_path, 'u1.base'), 
                                      delimiter='\t')
        self.training_set = np.array(self.training_set, dtype='int')
        
        self.test_set = pd.read_csv(os.path.join(ml100k_path, 'u1.test'), 
                                  delimiter='\t')
        self.test_set = np.array(self.test_set, dtype='int')
        
        print(f"Movies loaded: {len(self.movies)}")
        print(f"Users loaded: {len(self.users)}")
        print(f"Training ratings: {len(self.training_set)}")
        print(f"Test ratings: {len(self.test_set)}")
        
    def prepare_data(self):
        """Convert data into matrix format and binary ratings."""
        print("\nPreparing data matrices...")
        
        # Get number of users and movies
        self.nb_users = int(max(max(self.training_set[:, 0]), 
                               max(self.test_set[:, 0])))
        self.nb_movies = int(max(max(self.training_set[:, 1]), 
                                max(self.test_set[:, 1])))
        
        print(f"Number of users: {self.nb_users}")
        print(f"Number of movies: {self.nb_movies}")
        
        # Convert to list of lists format
        def convert(data):
            new_data = []
            for id_users in range(1, self.nb_users + 1):
                id_movies = data[:, 1][data[:, 0] == id_users]
                id_ratings = data[:, 2][data[:, 0] == id_users]
                ratings = np.zeros(self.nb_movies)
                ratings[id_movies - 1] = id_ratings
                new_data.append(list(ratings))
            return new_data
        
        self.training_set = convert(self.training_set)
        self.test_set = convert(self.test_set)
        
        # Convert to Torch tensors
        self.training_set = torch.FloatTensor(self.training_set)
        self.test_set = torch.FloatTensor(self.test_set)
        
        # Convert ratings to binary (liked/not liked)
        print("\nConverting to binary ratings...")
        self.training_set[self.training_set == 0] = -1
        self.training_set[self.training_set == 1] = 0
        self.training_set[self.training_set == 2] = 0
        self.training_set[self.training_set >= 3] = 1
        
        self.test_set[self.test_set == 0] = -1
        self.test_set[self.test_set == 1] = 0
        self.test_set[self.test_set == 2] = 0
        self.test_set[self.test_set >= 3] = 1
        
        print("Binary conversion: 1-2 → 0 (not liked), 3-5 → 1 (liked)")
        
    def build_model(self):
        """Initialize the RBM model."""
        print("\nBuilding RBM model...")
        nv = len(self.training_set[0])  # Number of visible units
        nh = 100  # Number of hidden units
        
        self.rbm = RBM(nv, nh)
        self.batch_size = 100
        
        print(f"Model architecture: {nv} visible units ↔ {nh} hidden units")
        print(f"Batch size: {self.batch_size}")
        
    def train(self, nb_epoch=10, k=10):
        """
        Train the RBM model.
        
        Args:
            nb_epoch: Number of training epochs
            k: Number of Gibbs sampling steps
        """
        print(f"\nStarting training for {nb_epoch} epochs...")
        print(f"Using CD-{k} (Contrastive Divergence with {k} steps)")
        
        start_time = time.time()
        train_losses = []
        
        for epoch in range(1, nb_epoch + 1):
            train_loss = 0
            s = 0.
            
            # Progress bar for batches
            pbar = tqdm(range(0, self.nb_users - self.batch_size, self.batch_size), 
                       desc=f'Epoch {epoch}/{nb_epoch}')
            
            for id_user in pbar:
                vk = self.training_set[id_user:id_user + self.batch_size]
                v0 = self.training_set[id_user:id_user + self.batch_size]
                ph0, _ = self.rbm.sample_h(v0)
                
                # Gibbs sampling
                for step in range(k):
                    _, hk = self.rbm.sample_h(vk)
                    _, vk = self.rbm.sample_v(hk)
                    vk[v0 < 0] = v0[v0 < 0]  # Keep unrated movies
                
                phk, _ = self.rbm.sample_h(vk)
                
                # Update weights
                self.rbm.train(v0, vk, ph0, phk)
                
                # Calculate loss
                train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
                s += 1.
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{train_loss/s:.4f}'})
            
            avg_loss = train_loss / s
            train_losses.append(avg_loss)
            print(f'Epoch {epoch}: Loss = {avg_loss:.4f}')
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Final training loss: {train_losses[-1]:.4f}")
        
        return train_losses
    
    def test(self):
        """Test the RBM model."""
        print("\nEvaluating on test set...")
        test_loss = 0
        s = 0.
        
        for id_user in tqdm(range(self.nb_users), desc='Testing'):
            v = self.training_set[id_user:id_user + 1]
            vt = self.test_set[id_user:id_user + 1]
            
            if len(vt[vt >= 0]) > 0:
                _, h = self.rbm.sample_h(v)
                _, v = self.rbm.sample_v(h)
                test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
                s += 1.
        
        avg_test_loss = test_loss / s
        print(f'Test Loss: {avg_test_loss:.4f}')
        
        return avg_test_loss
    
    def save_model(self, path='rbm_model.pth'):
        """Save the trained model."""
        torch.save({
            'W': self.rbm.W,
            'a': self.rbm.a,
            'b': self.rbm.b,
            'nb_movies': self.nb_movies,
            'nb_users': self.nb_users,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='rbm_model.pth'):
        """Load a trained model."""
        checkpoint = torch.load(path)
        nv = checkpoint['W'].shape[1]
        nh = checkpoint['W'].shape[0]
        
        self.rbm = RBM(nv, nh)
        self.rbm.W = checkpoint['W']
        self.rbm.a = checkpoint['a']
        self.rbm.b = checkpoint['b']
        self.nb_movies = checkpoint['nb_movies']
        self.nb_users = checkpoint['nb_users']
        
        print(f"Model loaded from {path}")
    
    def recommend_movies(self, user_id, top_n=10):
        """
        Get movie recommendations for a specific user.
        
        Returns movies predicted as "liked" (probability > 0.5)
        """
        v = self.training_set[user_id - 1:user_id]
        
        # Get predictions
        _, h = self.rbm.sample_h(v)
        p_v_given_h, _ = self.rbm.sample_v(h)
        
        # Get probabilities for unrated movies
        predictions = p_v_given_h.squeeze()
        unrated_mask = v.squeeze() < 0
        
        # Get top recommendations from unrated movies
        unrated_predictions = predictions[unrated_mask]
        unrated_indices = torch.where(unrated_mask)[0]
        
        # Sort by probability
        top_indices = torch.argsort(unrated_predictions, descending=True)[:top_n]
        
        recommendations = []
        for idx in top_indices:
            movie_idx = unrated_indices[idx].item()
            probability = unrated_predictions[idx].item()
            recommendations.append((movie_idx + 1, probability))
        
        return recommendations


def main():
    """Main training pipeline."""
    # Initialize recommender system
    recommender = MovieRecommenderRBM(data_path='data/')
    
    # Load and prepare data
    recommender.load_data()
    recommender.prepare_data()
    
    # Build and train model
    recommender.build_model()
    train_losses = recommender.train(nb_epoch=10, k=10)
    
    # Test model
    test_loss = recommender.test()
    
    # Save model
    recommender.save_model('rbm_model.pth')
    
    # Example: Get recommendations for user 1
    print("\nSample recommendations for User 1:")
    recommendations = recommender.recommend_movies(user_id=1, top_n=5)
    for movie_id, probability in recommendations:
        print(f"Movie {movie_id}: Like probability {probability:.3f}")


if __name__ == "__main__":
    main()