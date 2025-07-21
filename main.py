"""
Main script to run RBM experiment
Author: Ahmad Hammam
"""

from src.trainer import RBMExperiment


def main():
    """Run complete RBM training pipeline."""
    
    print("🎬 Restricted Boltzmann Machine - Movie Recommendation System")
    print("=" * 60)
    
    # Initialize experiment
    experiment = RBMExperiment(data_path="data/", model_save_path="models/")
    
    # Load and prepare data
    experiment.load_and_prepare_data(binary=True)
    
    # Initialize model with 100 hidden units
    experiment.initialize_model(n_hidden=100, batch_size=100)
    
    # Train model for 10 epochs with CD-10
    experiment.train_model(nb_epochs=10, k=10, save_checkpoints=True)
    
    # Evaluate model on test set
    experiment.evaluate_model()
    
    # Show sample recommendations
    experiment.demonstrate_recommendations(n_users=3, n_recommendations=10)
    
    # Save the trained model
    experiment.save_model("best_rbm_model.pth")
    
    # Plot training history
    experiment.plot_training_history()
    
    print("\n✅ Experiment completed successfully!")
    print(f"Model saved to: models/best_rbm_model.pth")


if __name__ == "__main__":
    main()