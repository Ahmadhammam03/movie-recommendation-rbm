# 🎬 Movie Recommendation System with Restricted Boltzmann Machines (RBM)

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://www.linkedin.com/in/ahmad-hammam-1561212b2)

**Binary movie recommendation system using Restricted Boltzmann Machines - predicts whether users will LIKE or DISLIKE movies with probabilistic deep learning.**

A probabilistic deep learning approach to movie recommendations using Restricted Boltzmann Machines (RBM) implemented in PyTorch. This project demonstrates energy-based models for collaborative filtering with **binary rating predictions** (like/dislike), perfect for thumbs-up/thumbs-down recommendation systems.

> 🎯 **Looking for rating predictions (1-5 stars)?** Check out my **[Stacked AutoEncoder implementation](https://github.com/Ahmadhammam03/movie-recommendation-sae)** for continuous rating predictions!

## 🌟 Features

- **Probabilistic Model**: RBM with Gibbs sampling for recommendation generation
- **Binary Classification**: Converts ratings to liked/not-liked predictions
- **Energy-Based Learning**: Unsupervised feature learning through energy minimization
- **Contrastive Divergence**: CD-k algorithm for efficient training
- **PyTorch Implementation**: Modern deep learning framework with GPU support

## 📊 Results

- **Training Loss**: ~0.247 after 10 epochs
- **Test Loss**: ~0.227 (excellent generalization)
- **Binary Accuracy**: High precision in like/dislike predictions
- **Architecture**: [nb_movies → 100 hidden units]
- **Training Time**: ~2 minutes for 10 epochs

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.7 or higher
python --version

# Install required packages
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Ahmadhammam03/movie-recommendation-rbm.git
cd movie-recommendation-rbm
```

2. Download the MovieLens datasets:
```bash
# Create data directory
mkdir -p data/ml-1m data/ml-100k

# Download datasets (or use provided links)
# ML-1M: https://grouplens.org/datasets/movielens/1m/
# ML-100K: https://grouplens.org/datasets/movielens/100k/
```

3. Run the training:
```bash
python train_rbm.py
```

## 📁 Project Structure

```
movie-recommendation-rbm/
├── data/
│   ├── README.md          # Dataset documentation
│   ├── ml-1m/
│   │   ├── movies.dat
│   │   ├── ratings.dat
│   │   └── users.dat
│   └── ml-100k/
│       ├── u1.base
│       ├── u1.test
│       └── ...
├── models/
│   ├── README.md          # Model storage documentation
│   ├── best_rbm_model.pth # Trained model (after running main.py)
│   └── checkpoints/       # Training checkpoints
├── notebooks/
│   └── rbm.ipynb          # Jupyter notebook implementation
├── src/
│   ├── __init__.py        # Package initialization
│   ├── model.py           # RBM model architecture
│   ├── data_loader.py     # Data preprocessing utilities
│   └── trainer.py         # Training logic & experiment class
├── main.py                # Main script to run full pipeline
├── test_recommendations.py # Script to test trained model
├── train_rbm.py           # Alternative training script
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

## 🔧 Model Architecture

### Restricted Boltzmann Machine

```python
Visible Layer: nb_movies neurons (binary ratings)
Hidden Layer: 100 neurons (learned features)
Weights: W (100 x nb_movies)
Visible Bias: b (1 x nb_movies)
Hidden Bias: a (1 x 100)
Activation: Sigmoid
Sampling: Gibbs sampling with k=10 steps
```

### Key Differences from Autoencoders:
- **Stochastic vs Deterministic**: RBMs use probabilistic sampling
- **Energy-Based**: Learns joint probability distribution
- **Binary Outputs**: Natural for like/dislike recommendations
- **Bidirectional**: Can generate both hidden and visible states

## 💻 Usage

### Basic Training

```python
from src.trainer import RBMExperiment

# Initialize experiment
experiment = RBMExperiment(data_path="data/", model_save_path="models/")

# Run complete pipeline
experiment.load_and_prepare_data(binary=True)
experiment.initialize_model(n_hidden=100)
experiment.train_model(nb_epochs=10)
experiment.evaluate_model()
experiment.save_model("best_rbm_model.pth")
```

### Quick Start with Main Script

```bash
# Train the model
python main.py

# Test recommendations
python test_recommendations.py
```

### Custom Configuration

```python
# Modify architecture
rbm = RBM(
    nv=nb_movies,
    nh=200,              # More hidden units
    k=15                 # More Gibbs sampling steps
)

# Different learning parameters
trainer = RBMTrainer(
    rbm,
    batch_size=50,
    learning_rate=0.01,
    momentum=0.9
)
```

## 📈 Training Details

### Hyperparameters
- **Hidden Units**: 100
- **Batch Size**: 100
- **Epochs**: 10
- **Gibbs Steps (k)**: 10
- **Learning Rate**: Implicit (weight updates)

### Binary Rating Conversion
```
Ratings 1-2 → 0 (Not Liked)
Ratings 3-5 → 1 (Liked)
Rating 0    → -1 (Not Rated)
```

### Loss Progression

| Epoch | Training Loss |
|-------|--------------|
| 1     | 0.3432       |
| 5     | 0.2471       |
| 10    | 0.2475       |

## 🎯 Key Algorithms

### Contrastive Divergence (CD-k)
1. **Positive Phase**: Sample hidden units from visible data
2. **Negative Phase**: Reconstruct visible units after k Gibbs steps
3. **Weight Update**: Difference between positive and negative statistics

### Gibbs Sampling
```python
# Sample hidden given visible
p(h|v) = sigmoid(v·W^T + a)
h ~ Bernoulli(p(h|v))

# Sample visible given hidden
p(v|h) = sigmoid(h·W + b)
v ~ Bernoulli(p(v|h))
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GroupLens](https://grouplens.org/) for providing the MovieLens dataset
- Geoffrey Hinton for pioneering work on RBMs and Deep Belief Networks
- PyTorch community for the excellent deep learning framework

## 👨‍💻 Author

**Ahmad Hammam**
- GitHub: [@Ahmadhammam03](https://github.com/Ahmadhammam03)
- LinkedIn: [Ahmad Hammam](https://www.linkedin.com/in/ahmad-hammam-1561212b2)

## 🔬 Theory Behind RBMs

### Energy Function
```
E(v,h) = -b^T·v - a^T·h - v^T·W·h
```

### Probability Distribution
```
P(v,h) = exp(-E(v,h)) / Z
```
Where Z is the partition function (normalization constant)

### Learning Rule
```
ΔW = ε(<v·h^T>_data - <v·h^T>_model)
```

## 📊 Comparison: RBM vs SAE

| Feature | RBM (This Project) | SAE ([Link](https://github.com/Ahmadhammam03/movie-recommendation-sae)) |
|---------|-----|-----|
| **Output Type** | **Binary (Like/Dislike)** | **Continuous (1-5 Stars)** |
| **Use Case** | **Thumbs Up/Down Systems** | **Rating Prediction** |
| **Model Type** | Generative (Energy-based) | Discriminative (Reconstruction) |
| **Learning** | Probabilistic Sampling | Deterministic Encoding |
| **Architecture** | Bipartite Graph | Multi-layer Network |
| **Training** | Contrastive Divergence | Backpropagation |
| **Best For** | Binary preferences, discovery | Precise rating prediction |

### When to Use Which:
- **🔥 RBM (This Project)**: Netflix-style thumbs up/down, Spotify-like discovery, binary feedback systems
- **⭐ SAE ([Other Project](https://github.com/Ahmadhammam03/movie-recommendation-sae))**: Amazon-style star ratings, detailed preference modeling, rating prediction

---

⭐ If you find this project useful, please consider giving it a star!
