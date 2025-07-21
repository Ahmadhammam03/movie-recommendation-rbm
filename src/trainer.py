"""
Training utilities for RBM
Author: Ahmad Hammam
"""

import torch
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from datetime import datetime
import os


class RBMTrainer:
    """
    Trainer class for Restricted Boltzmann Machine.
    """
    
    def __init__(self, rbm, batch_size=100, device='cpu'):
        """
        Initialize trainer.
        
        Args:
            rbm: RBM model instance
            batch_size: Training batch size
            device: Device to use ('cpu' or 'cuda')
        """
        self.rbm = rbm
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.train_losses = []
        self.test_losses = []
        
    def train_epoch(self, training_set, k=10):
        """
        Train for one epoch.
        
        Args:
            training_set: Training data
            k: Number of Gibbs sampling steps
            
        Returns:
            Average epoch loss
        """
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle users for each epoch
        n_users = len(training_set)
        user_order = torch.randperm(n_users)
        
        # Train in batches
        for batch_start in range(0, n_users - self.batch_size, self.batch_size):
            # Get batch indices
            batch_indices = user_order[batch_start:batch_start + self.batch_size]
            
            # Get batch data
            v0 = training_set[batch_indices]
            vk = v0.clone()
            
            # Positive phase
            ph0, _ = self.rbm.sample_h(v0)
            
            # Negative phase - k steps of Gibbs sampling
            for step in range(k):
                _, hk = self.rbm.sample_h(vk)
                _, vk = self.rbm.sample_v(hk)
                vk[v0 < 0] = v0[v0 < 0]  # Keep missing ratings
            
            # Final hidden sample
            phk, _ = self.rbm.sample_h(vk)
            
            # Update weights
            self.rbm.train(v0, vk, ph0, phk)
            
            # Calculate reconstruction error
            batch_loss = torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
            epoch_loss += batch_loss
            n_batches += 1
            
        return epoch_loss / n_batches
    
    def train(self, training_set, nb_epochs=10, k=10, verbose=True, 
              save_checkpoints=False, checkpoint_dir='checkpoints'):
        """
        Full training loop.
        
        Args:
            training_set: Training data
            nb_epochs: Number of epochs
            k: Number of Gibbs sampling steps
            verbose: Whether to print progress
            save_checkpoints: Whether to save model checkpoints
            checkpoint_dir: Directory to save checkpoints
        """
        if save_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        print(f"Starting RBM training for {nb_epochs} epochs...")
        print(f"Batch size: {self.batch_size}, CD-{k}")
        
        start_time = time.time()
        best_loss = float('inf')
        
        for epoch in range(1, nb_epochs + 1):
            # Train for one epoch
            if verbose:
                print(f"\nEpoch {epoch}/{nb_epochs}")
                
            epoch_loss = self.train_epoch(training_set, k)
            self.train_losses.append(epoch_loss.item())
            
            if verbose:
                print(f"Training Loss: {epoch_loss:.4f}")
                
            # Save checkpoint if improved
            if save_checkpoints and epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f'rbm_best.pth'),
                    epoch, epoch_loss
                )
                
            # Save periodic checkpoint
            if save_checkpoints and epoch % 10 == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f'rbm_epoch_{epoch}.pth'),
                    epoch, epoch_loss
                )
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Final training loss: {self.train_losses[-1]:.4f}")
        
    def test(self, training_set, test_set):
        """
        Evaluate model on test set.
        
        Args:
            training_set: Training data (for visible units)
            test_set: Test data (for evaluation)
            
        Returns:
            Average test loss
        """
        test_loss = 0
        s = 0.
        
        print("Evaluating on test set...")
        
        for id_user in tqdm(range(len(training_set))):
            v = training_set[id_user:id_user + 1]
            vt = test_set[id_user:id_user + 1]
            
            if len(vt[vt >= 0]) > 0:
                # Sample from RBM
                _, h = self.rbm.sample_h(v)
                _, v_pred = self.rbm.sample_v(h)
                
                # Calculate loss on test ratings
                test_loss += torch.mean(torch.abs(vt[vt >= 0] - v_pred[vt >= 0]))
                s += 1.
                
        avg_test_loss = test_loss / s
        self.test_losses.append(avg_test_loss.item())
        
        print(f"Test Loss: {avg_test_loss:.4f}")
        return avg_test_loss
    
    def save_checkpoint(self, filepath, epoch, loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state': {
                'W': self.rbm.W,
                'a': self.rbm.a,
                'b': self.rbm.b,
                'nv': self.rbm.nv,
                'nh': self.rbm.nh
            },
            'loss': loss,
            'train_losses': self.train_losses,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.rbm.W = checkpoint['model_state']['W']
        self.rbm.a = checkpoint['model_state']['a']
        self.rbm.b = checkpoint['model_state']['b']
        self.train_losses = checkpoint.get('train_losses', [])
        return checkpoint['epoch'], checkpoint['loss']
    
    def plot_losses(self, save_path=None):
        """Plot training and test losses."""
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        
        # Plot test losses if available
        if self.test_losses:
            test_epochs = np.linspace(1, len(self.train_losses), len(self.test_losses))
            plt.plot(test_epochs, self.test_losses, 'ro-', label='Test Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Error')
        plt.title('RBM Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_recommendations(self, user_ratings, n_recommendations=10):
        """
        Get movie recommendations for a user.
        
        Args:
            user_ratings: User's rating vector
            n_recommendations: Number of recommendations
            
        Returns:
            List of (movie_idx, probability) tuples
        """
        # Ensure input is tensor
        if not isinstance(user_ratings, torch.Tensor):
            user_ratings = torch.FloatTensor(user_ratings)
            
        # Add batch dimension if needed
        if user_ratings.dim() == 1:
            user_ratings = user_ratings.unsqueeze(0)
            
        # Get predictions
        _, h = self.rbm.sample_h(user_ratings)
        p_v_given_h, _ = self.rbm.sample_v(h)
        
        # Get probabilities for unrated movies
        predictions = p_v_given_h.squeeze()
        unrated_mask = user_ratings.squeeze() < 0
        
        # Filter to unrated movies only
        unrated_predictions = predictions[unrated_mask]
        unrated_indices = torch.where(unrated_mask)[0]
        
        # Get top recommendations
        top_values, top_indices = torch.topk(unrated_predictions, 
                                           min(n_recommendations, len(unrated_predictions)))
        
        recommendations = []
        for i in range(len(top_values)):
            movie_idx = unrated_indices[top_indices[i]].item()
            probability = top_values[i].item()
            recommendations.append((movie_idx, probability))
            
        return recommendations


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(self, patience=10, min_delta=0.0001):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """Check if should stop training."""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop


class ExperimentTracker:
    """
    Track and log experiments.
    """
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.metrics = {
            'train_losses': [],
            'test_losses': [],
            'epoch_times': [],
            'hyperparameters': {}
        }
        
    def log_hyperparameters(self, **kwargs):
        """Log hyperparameters."""
        self.metrics['hyperparameters'].update(kwargs)
        
    def log_epoch(self, epoch, train_loss, test_loss=None, epoch_time=None):
        """Log epoch metrics."""
        self.metrics['train_losses'].append(train_loss)
        if test_loss is not None:
            self.metrics['test_losses'].append(test_loss)
        if epoch_time is not None:
            self.metrics['epoch_times'].append(epoch_time)
            
    def save_results(self, filepath):
        """Save experiment results."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def print_summary(self):
        """Print experiment summary."""
        print(f"\nExperiment: {self.experiment_name}")
        print("Hyperparameters:")
        for key, value in self.metrics['hyperparameters'].items():
            print(f"  {key}: {value}")
        print(f"\nFinal train loss: {self.metrics['train_losses'][-1]:.4f}")
        if self.metrics['test_losses']:
            print(f"Final test loss: {self.metrics['test_losses'][-1]:.4f}")


class RBMExperiment:
    """
    High-level experiment class for RBM training and evaluation.
    """
    
    def __init__(self, data_path='data/', model_save_path='models/'):
        """
        Initialize RBM experiment.
        
        Args:
            data_path: Path to data directory
            model_save_path: Path to save models
        """
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.data_loader = MovieLensDataLoader(data_path=data_path)
        self.rbm = None
        self.trainer = None
        self.training_set = None
        self.test_set = None
        self.nb_users = None
        self.nb_movies = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)
        
    def load_and_prepare_data(self, binary=True):
        """
        Load and prepare data for RBM training.
        
        Args:
            binary: Whether to use binary ratings
        """
        print("Loading and preparing data...")
        self.data_loader.load_metadata()
        
        # Prepare RBM data
        self.training_set, self.test_set, self.nb_users, self.nb_movies = \
            self.data_loader.prepare_rbm_data(binary=binary)
        
        # Get and print statistics
        stats = self.data_loader.get_data_statistics(self.training_set, self.test_set)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
                
    def initialize_model(self, n_hidden=100, batch_size=100):
        """
        Initialize RBM model and trainer.
        
        Args:
            n_hidden: Number of hidden units
            batch_size: Training batch size
        """
        print(f"\nInitializing RBM model...")
        print(f"Architecture: {self.nb_movies} visible units ↔ {n_hidden} hidden units")
        
        # Initialize RBM
        from .model import RBM
        self.rbm = RBM(nv=self.nb_movies, nh=n_hidden)
        
        # Initialize trainer
        self.trainer = RBMTrainer(self.rbm, batch_size=batch_size)
        
    def train_model(self, nb_epochs=10, k=10, save_checkpoints=True):
        """
        Train the RBM model.
        
        Args:
            nb_epochs: Number of training epochs
            k: Number of Gibbs sampling steps
            save_checkpoints: Whether to save checkpoints
        """
        checkpoint_dir = os.path.join(self.model_save_path, 'checkpoints')
        
        self.trainer.train(
            self.training_set,
            nb_epochs=nb_epochs,
            k=k,
            save_checkpoints=save_checkpoints,
            checkpoint_dir=checkpoint_dir
        )
        
    def evaluate_model(self):
        """Evaluate model on test set."""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        test_loss = self.trainer.test(self.training_set, self.test_set)
        
        print(f"\nFinal Results:")
        print(f"  Training Loss: {self.trainer.train_losses[-1]:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        
        return test_loss
        
    def demonstrate_recommendations(self, n_users=3, n_recommendations=10):
        """
        Demonstrate recommendations for sample users.
        
        Args:
            n_users: Number of users to show
            n_recommendations: Number of recommendations per user
        """
        print("\n" + "="*50)
        print("SAMPLE RECOMMENDATIONS")
        print("="*50)
        
        for user_id in range(1, min(n_users + 1, self.nb_users + 1)):
            print(f"\n🎬 Top {n_recommendations} Recommendations for User {user_id}:")
            
            # Get user's ratings
            user_ratings = self.training_set[user_id - 1]
            
            # Get recommendations
            recommendations = self.trainer.get_recommendations(
                user_ratings, n_recommendations
            )
            
            # Display recommendations
            for i, (movie_idx, probability) in enumerate(recommendations, 1):
                movie_id = movie_idx + 1  # Convert to 1-indexed
                title, genres = self.data_loader.get_movie_info(movie_id)
                print(f"  {i:2d}. {title} ({genres})")
                print(f"      Probability of liking: {probability:.3f}")
                
    def save_model(self, filename):
        """Save the trained model."""
        filepath = os.path.join(self.model_save_path, filename)
        
        checkpoint = {
            'model_state': {
                'W': self.rbm.W,
                'a': self.rbm.a,
                'b': self.rbm.b,
                'nv': self.rbm.nv,
                'nh': self.rbm.nh
            },
            'nb_users': self.nb_users,
            'nb_movies': self.nb_movies,
            'train_losses': self.trainer.train_losses,
            'test_losses': self.trainer.test_losses,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        print(f"\nModel saved to: {filepath}")
        
    def load_model(self, filename):
        """Load a trained model."""
        filepath = os.path.join(self.model_save_path, filename)
        
        print(f"Loading model from: {filepath}")
        checkpoint = torch.load(filepath)
        
        # Restore dimensions
        self.nb_users = checkpoint['nb_users']
        self.nb_movies = checkpoint['nb_movies']
        
        # Initialize RBM with saved parameters
        from .model import RBM
        nv = checkpoint['model_state']['nv']
        nh = checkpoint['model_state']['nh']
        
        self.rbm = RBM(nv=nv, nh=nh)
        self.rbm.W = checkpoint['model_state']['W']
        self.rbm.a = checkpoint['model_state']['a']
        self.rbm.b = checkpoint['model_state']['b']
        
        # Initialize trainer
        self.trainer = RBMTrainer(self.rbm)
        self.trainer.train_losses = checkpoint.get('train_losses', [])
        self.trainer.test_losses = checkpoint.get('test_losses', [])
        
        print(f"Model loaded successfully!")
        print(f"Architecture: {nv} visible units ↔ {nh} hidden units")
        
    def plot_training_history(self):
        """Plot training history."""
        if self.trainer:
            save_path = os.path.join(self.model_save_path, 'training_history.png')
            self.trainer.plot_losses(save_path=save_path)


# Example usage
if __name__ == "__main__":
    # Create experiment
    experiment = RBMExperiment(data_path='../data/', model_save_path='../models/')
    
    # Run complete pipeline
    experiment.load_and_prepare_data(binary=True)
    experiment.initialize_model(n_hidden=100)
    experiment.train_model(nb_epochs=10)
    experiment.evaluate_model()
    experiment.demonstrate_recommendations()
    experiment.save_model("best_rbm_model.pth")