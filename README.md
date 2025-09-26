We’ll use the TMDB 5000 Movies dataset from Kaggle, but I’ll also make it flexible so you can swap datasets later.

How It Works:<br>
1.Takes a movie title (like Avatar).<br>
2.Finds its content vector (overview + genres + keywords).<br>
3.Computes cosine similarity with all other movies.<br>
4.Returns Top-N most similar movies.
