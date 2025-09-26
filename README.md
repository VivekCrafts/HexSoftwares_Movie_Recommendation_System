We’ll use the TMDB 5000 Movies dataset from Kaggle, but I’ll also make it flexible so you can swap datasets later.

How It Works:
Takes a movie title (like Avatar).
Finds its content vector (overview + genres + keywords).
Computes cosine similarity with all other movies.
Returns Top-N most similar movies.
