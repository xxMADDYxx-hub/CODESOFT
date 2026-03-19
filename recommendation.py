# ============================================================
# TASK 4: RECOMMENDATION SYSTEM
# CodSoft AI Internship
#
# Requirements:
#   pip install pandas numpy scikit-learn
#
# Includes:
#   - Content-Based Filtering (by genre/description)
#   - Collaborative Filtering (by user ratings)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Sample Dataset (100 Movies) ----------
MOVIES = pd.DataFrame({
    'title': [
        # Sci-Fi / Action
        'The Matrix', 'Inception', 'Interstellar', 'The Dark Knight',
        'Avengers: Endgame', 'Iron Man', 'Doctor Strange', 'Guardians of the Galaxy',
        'Avatar', 'Alien', 'Blade Runner 2049', 'Edge of Tomorrow',
        'Minority Report', 'The Martian', 'Gravity', 'Arrival',
        'Thor: Ragnarok', 'Black Panther', 'Spider-Man: Into the Spider-Verse', 'Mad Max: Fury Road',
        # Crime / Drama
        'The Shawshank Redemption', 'Forrest Gump', 'The Godfather', "Schindler's List",
        'Goodfellas', 'The Silence of the Lambs', 'Se7en', 'Fight Club',
        'The Departed', 'No Country for Old Men', 'Prisoners', 'Zodiac',
        'Heat', 'Catch Me If You Can', 'The Green Mile', 'American History X',
        # Animation / Family
        'Toy Story', 'Finding Nemo', 'The Lion King', 'Frozen',
        'Up', 'WALL-E', 'Inside Out', 'Coco',
        'Moana', 'Zootopia', 'Shrek', 'How to Train Your Dragon',
        'Kung Fu Panda', 'The Incredibles', 'Ratatouille', 'Spirited Away',
        # Romance / Drama
        'Titanic', 'The Notebook', 'Pride & Prejudice', 'La La Land',
        'A Beautiful Mind', 'Good Will Hunting', 'Dead Poets Society', 'The Theory of Everything',
        'Atonement', 'Brooklyn', 'Her', 'Eternal Sunshine of the Spotless Mind',
        'Before Sunrise', 'Silver Linings Playbook', 'Crazy Rich Asians', 'Me Before You',
        # Thriller / Mystery
        'Gone Girl', 'Shutter Island', 'Knives Out', 'The Prestige',
        'Memento', 'Rear Window', 'Parasite', 'Oldboy',
        'The Sixth Sense', 'Psycho', 'Get Out', 'Us',
        'A Quiet Place', 'The Others', 'Hereditary', 'Black Swan',
        # History / War / Biography
        'Gladiator', 'Braveheart', 'Saving Private Ryan', 'Dunkirk',
        '1917', 'The Pianist', 'Hacksaw Ridge', 'Apocalypse Now',
        'Lawrence of Arabia', 'Patton', 'Full Metal Jacket', 'Platoon',
        # Comedy
        'The Grand Budapest Hotel', 'Superbad', 'The Big Lebowski', 'Groundhog Day',
        'Home Alone', 'Mrs. Doubtfire', 'Ferris Bueller\'s Day Off', 'Monty Python and the Holy Grail',
        # Horror
        'The Shining', 'It', 'A Nightmare on Elm Street', 'The Conjuring',
    ],
    'genre': [
        # Sci-Fi / Action
        'sci-fi action thriller', 'sci-fi thriller mystery', 'sci-fi drama adventure', 'action crime thriller',
        'action sci-fi superhero', 'action sci-fi superhero', 'action sci-fi fantasy superhero', 'action comedy sci-fi superhero',
        'sci-fi action adventure', 'sci-fi horror thriller', 'sci-fi thriller drama', 'sci-fi action thriller',
        'sci-fi thriller mystery', 'sci-fi drama adventure', 'sci-fi thriller drama', 'sci-fi drama mystery',
        'action comedy fantasy superhero', 'action drama sci-fi superhero', 'animation action sci-fi superhero', 'action sci-fi thriller',
        # Crime / Drama
        'drama crime', 'drama romance comedy', 'crime drama', 'drama history war',
        'crime drama biography', 'crime thriller drama', 'crime thriller mystery', 'drama thriller',
        'crime drama thriller', 'crime drama thriller', 'crime drama mystery thriller', 'crime mystery thriller',
        'crime drama action thriller', 'biography crime drama', 'drama fantasy crime', 'drama history',
        # Animation / Family
        'animation comedy adventure family', 'animation comedy adventure family', 'animation drama family', 'animation comedy fantasy family',
        'animation drama adventure family', 'animation sci-fi drama family', 'animation comedy drama family', 'animation comedy adventure family',
        'animation comedy adventure family', 'animation comedy adventure family', 'animation comedy fantasy adventure', 'animation comedy adventure family',
        'animation comedy action adventure family', 'animation comedy action adventure family', 'animation comedy adventure family', 'animation adventure fantasy family',
        # Romance / Drama
        'romance drama history', 'romance drama', 'romance drama', 'romance drama music',
        'biography drama romance', 'drama romance', 'drama romance', 'biography drama romance',
        'romance drama war', 'romance drama', 'romance drama sci-fi', 'romance drama sci-fi',
        'romance drama', 'romance comedy drama', 'romance comedy drama', 'romance drama',
        # Thriller / Mystery
        'thriller mystery drama', 'thriller mystery drama', 'comedy mystery thriller', 'thriller mystery sci-fi',
        'thriller mystery', 'thriller mystery', 'thriller drama mystery', 'thriller mystery drama',
        'thriller mystery drama', 'thriller horror mystery', 'horror thriller mystery', 'horror thriller mystery',
        'horror thriller mystery', 'horror thriller mystery', 'horror drama mystery', 'thriller drama horror',
        # History / War / Biography
        'action drama history', 'action drama history war', 'drama history war action', 'drama history war thriller',
        'drama history war thriller', 'biography drama history war', 'biography drama history war', 'drama war history',
        'biography drama history war', 'biography drama history war', 'drama history war', 'drama history war',
        # Comedy
        'comedy drama romance', 'comedy', 'comedy crime', 'comedy fantasy drama',
        'comedy family', 'comedy drama family', 'comedy drama', 'comedy fantasy',
        # Horror
        'horror thriller drama', 'horror thriller', 'horror thriller', 'horror thriller mystery',
    ],
    'rating': [
        # Sci-Fi / Action
        8.7, 8.8, 8.6, 9.0,
        8.4, 7.9, 7.5, 8.0,
        7.9, 8.4, 8.0, 7.9,
        7.6, 8.0, 7.7, 7.9,
        7.9, 7.3, 8.4, 8.1,
        # Crime / Drama
        9.3, 8.8, 9.2, 9.0,
        8.7, 8.6, 8.6, 8.8,
        8.5, 8.2, 8.1, 7.8,
        8.2, 8.1, 8.6, 8.5,
        # Animation / Family
        8.3, 8.1, 8.5, 7.4,
        8.2, 8.4, 8.1, 8.4,
        7.6, 8.0, 7.8, 8.1,
        7.6, 8.0, 8.1, 8.6,
        # Romance / Drama
        7.8, 7.9, 7.8, 8.0,
        8.2, 8.3, 8.1, 7.7,
        7.8, 7.4, 8.0, 8.3,
        8.1, 7.8, 6.9, 7.4,
        # Thriller / Mystery
        8.1, 8.1, 7.9, 8.5,
        8.4, 8.5, 8.5, 8.4,
        8.1, 8.5, 7.7, 6.8,
        7.5, 7.6, 7.3, 8.0,
        # History / War / Biography
        8.5, 8.3, 8.6, 7.9,
        8.3, 8.5, 8.1, 8.4,
        8.3, 7.9, 8.3, 8.1,
        # Comedy
        8.1, 7.6, 8.1, 8.0,
        7.7, 6.9, 7.8, 8.2,
        # Horror
        8.4, 6.9, 6.5, 7.5,
    ]
})

# ---------- User Rating Matrix (for Collaborative Filtering) ----------
USER_RATINGS = pd.DataFrame({
    'The Matrix':               [5, 4, 0, 0, 3, 5, 0],
    'Inception':                [5, 5, 0, 0, 0, 4, 0],
    'Interstellar':             [4, 5, 0, 0, 0, 3, 0],
    'The Dark Knight':          [4, 4, 0, 0, 2, 5, 0],
    'Avengers: Endgame':        [3, 0, 5, 4, 0, 0, 0],
    'Iron Man':                 [0, 0, 5, 5, 0, 0, 0],
    'Toy Story':                [0, 0, 0, 3, 5, 0, 4],
    'Finding Nemo':             [0, 0, 0, 4, 5, 0, 5],
    'Titanic':                  [0, 0, 0, 0, 0, 0, 4],
    'The Notebook':             [0, 0, 0, 0, 0, 0, 5],
    'The Shawshank Redemption': [3, 4, 0, 5, 0, 4, 0],
    'Forrest Gump':             [0, 3, 0, 5, 4, 0, 3],
}, index=['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Frank', 'Grace'])

# ---------- Content-Based Filtering ----------
def content_based_recommend(movie_title: str, top_n: int = 5):
    """Recommend movies similar to the given title based on genre similarity."""
    if movie_title not in MOVIES['title'].values:
        return None, f"Movie '{movie_title}' not found in the database."

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(MOVIES['genre'])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = MOVIES[MOVIES['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]

    recommendations = MOVIES.iloc[[s[0] for s in scores]][['title', 'genre', 'rating']]
    return recommendations, None

# ---------- Collaborative Filtering ----------
def collaborative_recommend(username: str, top_n: int = 5):
    """Recommend movies to a user based on similar users' ratings."""
    if username not in USER_RATINGS.index:
        return None, f"User '{username}' not found."

    # Compute user similarity
    user_sim = cosine_similarity(USER_RATINGS.fillna(0))
    user_sim_df = pd.DataFrame(user_sim, index=USER_RATINGS.index, columns=USER_RATINGS.index)

    # Get most similar users (excluding self)
    similar_users = user_sim_df[username].drop(username).sort_values(ascending=False)

    # Find movies the user hasn't rated
    user_ratings = USER_RATINGS.loc[username]
    unrated_movies = user_ratings[user_ratings == 0].index.tolist()

    if not unrated_movies:
        return None, f"{username} has already rated all movies."

    # Weighted average rating from similar users
    scores = {}
    for movie in unrated_movies:
        weighted_sum = 0
        sim_sum = 0
        for other_user, sim_score in similar_users.items():
            rating = USER_RATINGS.loc[other_user, movie]
            if rating > 0:
                weighted_sum += sim_score * rating
                sim_sum += sim_score
        if sim_sum > 0:
            scores[movie] = weighted_sum / sim_sum

    if not scores:
        return None, "Not enough data to make recommendations."

    top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result = pd.DataFrame(top_movies, columns=['title', 'predicted_rating'])
    result['predicted_rating'] = result['predicted_rating'].round(2)
    result = result.merge(MOVIES[['title', 'rating']], on='title', how='left')
    result = result.rename(columns={'rating': 'imdb_rating'})
    return result, None

# ---------- Resolve Movie by Number or Name ----------
def resolve_movie(user_input: str):
    """Returns a movie row given either a serial number or movie name. Returns (row, error)."""
    user_input = user_input.strip()

    # Try as serial number first
    if user_input.isdigit():
        num = int(user_input)
        if 1 <= num <= len(MOVIES):
            return MOVIES.iloc[num - 1], None
        else:
            return None, f"Number {num} is out of range. Please enter 1 to {len(MOVIES)}."

    # Try as movie name (case-insensitive)
    match = MOVIES[MOVIES['title'].str.lower() == user_input.lower()]
    if not match.empty:
        return match.iloc[0], None

    return None, f"'{user_input}' not found. Enter a number (1-{len(MOVIES)}) or exact movie name."

# ---------- Display Helpers ----------
def list_movies():
    print("\nAvailable Movies:")
    print(f"  {'#':>2}  {'Title':<35} {'IMDb Rating':>11}")
    print("  " + "-" * 50)
    for i, row in enumerate(MOVIES.itertuples(), 1):
        print(f"  {i:2}. {row.title:<35} ⭐ {row.rating}")

def list_users():
    print("\nAvailable Users:", ', '.join(USER_RATINGS.index.tolist()))

# ---------- Main Menu ----------
def main():
    print("=" * 55)
    print("     RECOMMENDATION SYSTEM  -  CodSoft AI Task 4")
    print("=" * 55)

    while True:
        print("\nOptions:")
        print("  1. Content-Based Recommendations (by movie)")
        print("  2. Collaborative Filtering (by user)")
        print("  3. List all movies")
        print("  4. List all users")
        print("  5. Exit")

        choice = input("\nChoose (1-5): ").strip()

        if choice == '1':
            list_movies()
            user_input = input("\nEnter movie number or name: ").strip()
            row, error = resolve_movie(user_input)
            if error:
                print(f"❌ {error}")
            else:
                print(f"\n🎬 {row['title']}")
                print(f"   Genre  : {row['genre']}")
                print(f"   Rating : ⭐ {row['rating']} / 10")

        elif choice == '2':
            list_users()
            user = input("\nEnter username: ").strip()
            results, error = collaborative_recommend(user)
            if error:
                print(f"Error: {error}")
            else:
                print(f"\n🎯 Recommended movies for '{user}':")
                print(results.to_string(index=False))

        elif choice == '3':
            list_movies()
            user_input = input("\nEnter movie number or name: ").strip()
            row, error = resolve_movie(user_input)
            if error:
                print(f"❌ {error}")
            else:
                print(f"\n🎬 {row['title']}")
                print(f"   Genre  : {row['genre']}")
                print(f"   Rating : ⭐ {row['rating']} / 10")

        elif choice == '4':
            list_users()

        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1–5.")


if __name__ == "__main__":
    main()