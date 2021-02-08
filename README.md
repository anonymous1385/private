# Enabling Users to Build Their Own Fair Recommender Systems without Log Data

## Dependency

```
$ pip install implicit
```

## Data

You can download and preprocess data by the following command. It may take time.
```
$ bash data.sh
```

`hetrec.npy` is the Last.fm dataset. `home_and_kitchen.npy` is the Amazon dataset. `adult_*.npy` and `adult_*.npz` are the Adult dataset.

Note: 404 error happended in the MovieLens server https://grouplens.org/datasets/movielens/ at the time of submission, though it was available one month before. If the link is still invalid, please download them elsewhere (or use your files if you have already downloaded it before). Or you can still use the Amazon and Adult dataset.


## How to use

```
$ python ml.py 100k cosine popularity 
$ python ml.py 100k bpr popularity 
$ python ml.py 100k cosine old
$ python ml.py 100k bpr old
$ python ml.py hetrec bpr popularity
$ python ml.py home bpr popularity
$ python adult.py
```

`100k` is the MovieLens 100k dataset. `hetrec` is the LastFM dataset. `home` is the Amazon Home and Kitchen dataset.
