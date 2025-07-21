"""
Test recommendations from trained RBM model
Author: Ahmad Hammam
"""

from src.trainer import RBMExperiment
import torch


def test_specific_user(experiment, user_id, n_recommendations=10):
    """
    Get recommendations for a specific user.
    
    Args:
        experiment: RBMExperiment instance
        user_id: User ID (1-indexed)
        n_recommendations: Number of recommendations
    """
    print(f"\n🎬 Top {n_recommendations} Movie Recommendations for User {user_id}:")
    print("-" * 60)
    
    # Get user's ratings (convert to 0-indexed)
    user_ratings = experiment.training_set[user_id - 1]
    
    # Count user's existing ratings
    n_rated = (user_ratings >= 0).sum().item()
    n_liked = (user_ratings == 1).sum().item()
    n_disliked = (user_ratings == 0).sum().item()
    
    print(f"User {user_id} Statistics:")
    print(f"  - Total movies rated: {n_rated}")
    print(f"  - Movies liked (3-5 stars): {n_liked}")
    print(f"  - Movies disliked (1-2 stars): {n_disliked}")
    print(f"\nRecommendations (unrated movies predicted as 'liked'):\n")
    
    # Get recommendations
    recommendations = experiment.trainer.get_recommendations(
        user_ratings, n_recommendations
    )
    
    # Display recommendations
    for i, (movie_idx, probability) in enumerate(recommendations, 1):
        movie_id = movie_idx + 1  # Convert to 1-indexed
        
        # Get movie info if available
        title, genres = experiment.data_loader.get_movie_info(movie_id)
        
        print(f"  {i:2d}. Movie ID {movie_id:4d}")
        print(f"      Title: {title}")
        print(f"      Genres: {genres}")
        print(f"      Probability of liking: {probability:.3f}")
        print()


def compare_users(experiment, user_ids, n_recommendations=5):
    """
    Compare recommendations for multiple users.
    
    Args:
        experiment: RBMExperiment instance
        user_ids: List of user IDs to compare
        n_recommendations: Number of recommendations per user
    """
    print("\n📊 Comparing Recommendations Across Users")
    print("=" * 80)
    
    all_recommendations = {}
    
    for user_id in user_ids:
        user_ratings = experiment.training_set[user_id - 1]
        recommendations = experiment.trainer.get_recommendations(
            user_ratings, n_recommendations * 2  # Get extra for overlap analysis
        )
        all_recommendations[user_id] = recommendations
        
        print(f"\nUser {user_id} - Top {n_recommendations} recommendations:")
        for i, (movie_idx, prob) in enumerate(recommendations[:n_recommendations], 1):
            movie_id = movie_idx + 1
            title, _ = experiment.data_loader.get_movie_info(movie_id)
            print(f"  {i}. {title[:50]:50s} (p={prob:.3f})")
    
    # Find overlapping recommendations
    print("\n🔍 Overlapping Recommendations:")
    movie_counts = {}
    for user_id, recs in all_recommendations.items():
        for movie_idx, _ in recs[:n_recommendations]:
            movie_id = movie_idx + 1
            if movie_id not in movie_counts:
                movie_counts[movie_id] = []
            movie_counts[movie_id].append(user_id)
    
    overlaps = {k: v for k, v in movie_counts.items() if len(v) > 1}
    if overlaps:
        for movie_id, users in overlaps.items():
            title, _ = experiment.data_loader.get_movie_info(movie_id)
            print(f"  - {title}: Recommended to users {users}")
    else:
        print("  No overlapping recommendations found.")


def analyze_genre_preferences(experiment, user_id):
    """
    Analyze genre preferences based on user's ratings and recommendations.
    
    Args:
        experiment: RBMExperiment instance
        user_id: User ID to analyze
    """
    print(f"\n🎭 Genre Analysis for User {user_id}")
    print("-" * 60)
    
    user_ratings = experiment.training_set[user_id - 1]
    
    # Analyze liked movies
    liked_genres = {}
    for idx in range(len(user_ratings)):
        if user_ratings[idx] == 1:  # Liked movie
            movie_id = idx + 1
            _, genres = experiment.data_loader.get_movie_info(movie_id)
            for genre in genres.split('|'):
                liked_genres[genre] = liked_genres.get(genre, 0) + 1
    
    if liked_genres:
        print("Genres from liked movies:")
        sorted_genres = sorted(liked_genres.items(), key=lambda x: x[1], reverse=True)
        for genre, count in sorted_genres[:10]:
            print(f"  - {genre}: {count} movies")
    
    # Analyze recommended movies
    recommendations = experiment.trainer.get_recommendations(user_ratings, 20)
    rec_genres = {}
    for movie_idx, _ in recommendations:
        movie_id = movie_idx + 1
        _, genres = experiment.data_loader.get_movie_info(movie_id)
        for genre in genres.split('|'):
            rec_genres[genre] = rec_genres.get(genre, 0) + 1
    
    print("\nGenres in top 20 recommendations:")
    sorted_rec_genres = sorted(rec_genres.items(), key=lambda x: x[1], reverse=True)
    for genre, count in sorted_rec_genres[:10]:
        print(f"  - {genre}: {count} movies")


def main():
    """Main function to test recommendations."""
    
    print("🎬 RBM Movie Recommendation Testing")
    print("=" * 80)
    
    # Load trained model
    experiment = RBMExperiment(data_path="data/", model_save_path="models/")
    
    try:
        experiment.load_model("best_rbm_model.pth")
    except FileNotFoundError:
        print("❌ Model file not found! Please run main.py first to train the model.")
        return
    
    # Load data for testing
    experiment.load_and_prepare_data(binary=True)
    
    # Test recommendations for specific users
    test_users = [1, 10, 100]
    for user_id in test_users:
        test_specific_user(experiment, user_id, n_recommendations=10)
    
    # Compare recommendations across users
    compare_users(experiment, user_ids=[1, 2, 3], n_recommendations=5)
    
    # Analyze genre preferences
    analyze_genre_preferences(experiment, user_id=1)
    
    # Interactive mode
    print("\n" + "="*80)
    print("Interactive Mode - Enter a user ID to get recommendations (or 'q' to quit)")
    print("="*80)
    
    while True:
        try:
            user_input = input("\nEnter user ID (1-{}) or 'q' to quit: ".format(experiment.nb_users))
            
            if user_input.lower() == 'q':
                break
                
            user_id = int(user_input)
            
            if 1 <= user_id <= experiment.nb_users:
                test_specific_user(experiment, user_id, n_recommendations=10)
            else:
                print(f"Invalid user ID. Please enter a number between 1 and {experiment.nb_users}")
                
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
    
    print("\n✅ Testing completed!")


if __name__ == "__main__":
    main()