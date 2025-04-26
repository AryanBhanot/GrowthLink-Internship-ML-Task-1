import pandas as pd

def load_training_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if len(parts) == 4:
                movie_id, title, genre, plot = parts
                data.append({
                    'id': int(movie_id),
                    'title': title,
                    'genre': genre,
                    'plot': plot
                })
    return pd.DataFrame(data)
