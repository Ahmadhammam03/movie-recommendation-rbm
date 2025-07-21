"""
RBM Model Architecture
Author: Ahmad Hammam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBM:
    """
    Restricted Boltzmann Machine for Collaborative Filtering
    
    An energy-based probabilistic model that learns a joint probability
    distribution over visible (movies) and hidden (features) units.
    """
    
    def __init__(self, nv, nh, k=1, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        """
        Initialize RBM.
        
        Args:
            nv (int): Number of visible units (movies)
            nh (int): Number of hidden units
            k (int): Number of Gibbs sampling steps
            learning_rate (float): Learning rate for weight updates
            momentum (float): Momentum coefficient
            weight_decay (float): L2 regularization coefficient
        """
        self.nv = nv
        self.nh = nh
        self.k = k
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize parameters
        self.W = torch.randn(nh, nv) * 0.01  # Small random weights
        self.a = torch.zeros(1, nh)           # Hidden bias
        self.b = torch.zeros(1, nv)           # Visible bias
        
        # Momentum terms
        self.W_momentum = torch.zeros_like(self.W)
        self.a_momentum = torch.zeros_like(self.a)
        self.b_momentum = torch.zeros_like(self.b)
        
    def sample_h(self, x):
        """
        Sample hidden units given visible units.
        P(h=1|v) = sigmoid(v*W^T + a)
        """
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        """
        Sample visible units given hidden units.
        P(v=1|h) = sigmoid(h*W + b)
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
        # Compute gradients
        positive_grad = torch.mm(v0.t(), ph0).t()
        negative_grad = torch.mm(vk.t(), phk).t()
        
        # Update weights with momentum
        self.W_momentum = self.momentum * self.W_momentum + \
                         self.learning_rate * (positive_grad - negative_grad - self.weight_decay * self.W)
        self.W += self.W_momentum
        
        # Update visible bias
        self.b_momentum = self.momentum * self.b_momentum + \
                         self.learning_rate * torch.sum((v0 - vk), 0)
        self.b += self.b_momentum
        
        # Update hidden bias
        self.a_momentum = self.momentum * self.a_momentum + \
                         self.learning_rate * torch.sum((ph0 - phk), 0)
        self.a += self.a_momentum
    
    def contrastive_divergence(self, v0, k=None):
        """
        Perform k steps of Contrastive Divergence.
        
        Args:
            v0: Initial visible state
            k: Number of Gibbs sampling steps (uses self.k if None)
            
        Returns:
            Tuple of (vk, ph0, phk) for weight updates
        """
        if k is None:
            k = self.k
            
        # Positive phase
        ph0, _ = self.sample_h(v0)
        
        # Negative phase - k steps of Gibbs sampling
        vk = v0.clone()
        for _ in range(k):
            _, hk = self.sample_h(vk)
            _, vk = self.sample_v(hk)
            # Keep original values for missing ratings
            vk[v0 < 0] = v0[v0 < 0]
        
        # Final hidden probabilities
        phk, _ = self.sample_h(vk)
        
        return vk, ph0, phk
    
    def free_energy(self, v):
        """
        Compute free energy of visible configuration.
        F(v) = -b'v - sum(log(1 + exp(W*v + a)))
        """
        wx_b = torch.mm(v, self.W.t()) + self.a.expand_as(torch.mm(v, self.W.t()))
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), dim=1)
        visible_term = torch.mm(v, self.b.t()).squeeze()
        return -visible_term - hidden_term


class DeepBeliefNetwork:
    """
    Deep Belief Network - Stack of RBMs
    
    Pre-train layer by layer using RBMs, then fine-tune
    with backpropagation.
    """
    
    def __init__(self, layer_sizes, k=1):
        """
        Initialize DBN.
        
        Args:
            layer_sizes: List of layer sizes [visible, hidden1, hidden2, ...]
            k: Number of Gibbs sampling steps for each RBM
        """
        self.rbm_layers = []
        self.n_layers = len(layer_sizes) - 1
        
        # Create RBM for each layer
        for i in range(self.n_layers):
            rbm = RBM(
                nv=layer_sizes[i],
                nh=layer_sizes[i + 1],
                k=k
            )
            self.rbm_layers.append(rbm)
    
    def pretrain(self, X, epochs=10, batch_size=10):
        """
        Layer-wise pre-training of DBN.
        
        Args:
            X: Input data
            epochs: Number of epochs per layer
            batch_size: Batch size for training
        """
        input_data = X
        
        for i, rbm in enumerate(self.rbm_layers):
            print(f"Pre-training RBM layer {i+1}/{self.n_layers}")
            
            for epoch in range(epochs):
                epoch_loss = 0
                n_batches = 0
                
                # Mini-batch training
                for batch_start in range(0, len(input_data) - batch_size, batch_size):
                    batch = input_data[batch_start:batch_start + batch_size]
                    
                    # Contrastive divergence
                    vk, ph0, phk = rbm.contrastive_divergence(batch)
                    rbm.train(batch, vk, ph0, phk)
                    
                    # Compute reconstruction error
                    epoch_loss += torch.mean(torch.abs(batch[batch >= 0] - vk[batch >= 0]))
                    n_batches += 1
                
                print(f"  Epoch {epoch+1}: Loss = {epoch_loss/n_batches:.4f}")
            
            # Transform data for next layer
            with torch.no_grad():
                input_data, _ = rbm.sample_h(input_data)
    
    def transform(self, X):
        """
        Transform data through all RBM layers.
        
        Args:
            X: Input data
            
        Returns:
            Transformed representation
        """
        output = X
        for rbm in self.rbm_layers:
            output, _ = rbm.sample_h(output)
        return output
    
    def reconstruct(self, X):
        """
        Reconstruct data by forward and backward pass.
        
        Args:
            X: Input data
            
        Returns:
            Reconstructed data
        """
        # Forward pass
        hidden_states = [X]
        current = X
        
        for rbm in self.rbm_layers:
            current, _ = rbm.sample_h(current)
            hidden_states.append(current)
        
        # Backward pass
        for i in range(len(self.rbm_layers) - 1, -1, -1):
            current, _ = self.rbm_layers[i].sample_v(current)
        
        return current


class ConditionalRBM:
    """
    Conditional RBM that can incorporate user features.
    
    Extends basic RBM to condition on additional user information
    like demographics, past behavior, etc.
    """
    
    def __init__(self, nv, nh, n_features):
        """
        Initialize Conditional RBM.
        
        Args:
            nv: Number of visible units (movies)
            nh: Number of hidden units
            n_features: Number of user features
        """
        self.nv = nv
        self.nh = nh
        self.n_features = n_features
        
        # Standard RBM weights
        self.W = torch.randn(nh, nv) * 0.01
        self.a = torch.zeros(1, nh)
        self.b = torch.zeros(1, nv)
        
        # Additional weights for user features
        self.U = torch.randn(nh, n_features) * 0.01  # Features to hidden
        self.V = torch.randn(nv, n_features) * 0.01  # Features to visible
    
    def sample_h(self, x, features):
        """
        Sample hidden units given visible units and features.
        """
        wx = torch.mm(x, self.W.t())
        uf = torch.mm(features, self.U.t())
        activation = wx + uf + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y, features):
        """
        Sample visible units given hidden units and features.
        """
        wy = torch.mm(y, self.W)
        vf = torch.mm(features, self.V.t())
        activation = wy + vf + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)