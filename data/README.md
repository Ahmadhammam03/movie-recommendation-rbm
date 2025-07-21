# Dataset Instructions

This project uses MovieLens datasets for training the Restricted Boltzmann Machine (RBM) recommendation system.

## 📊 Current Data Structure

Based on your uploaded files, you have:

### ML-1M Dataset (data/ml-1m/):
- `movies.dat` - Movie information
- `ratings.dat` - Rating information  
- `users.dat` - User demographics

### ML-100K Dataset (data/ml-100k/):
- `u1.base` - Training set (100k format)
- `u1.test` - Test set (100k format)
- Various other utility files

## 🔧 Data Format

### Original DAT Files Format:
The `.dat` files use `::` as separator:

**movies.dat**:
```
MovieID::Title::Genres
1::Toy Story (1995)::Animation|Children's|Comedy
```

**users.dat**:
```
UserID::Gender::Age::Occupation::Zip-code
1::F::1::10::48067
```

**ratings.dat**:
```
UserID::MovieID::Rating::Timestamp
1::1193::5::978300760
```

### Training/Test Files Format:
The `u1.base` and `u1.test` files use tab separation:
```
UserID  MovieID  Rating  Timestamp
1       2       3       876893171
```

## 🚀 Usage in Code

The RBM project automatically loads and processes these files:

```python
# Loading metadata
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Loading training/test sets
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
```

## 📁 File Structure

```
data/
├── ml-1m/
│   ├── movies.dat           # 3,883 movies
│   ├── ratings.dat          # 1,000,209 ratings
│   └── users.dat            # 6,040 users
└── ml-100k/
    ├── u1.base              # 80,000 ratings (training)
    ├── u1.test              # 20,000 ratings (test)
    └── ... (other splits)
```

## 🔍 Binary Rating Conversion

For RBM training, ratings are converted to binary format:

- **Ratings 1-2**: → 0 (Not Liked)
- **Ratings 3-5**: → 1 (Liked)  
- **No Rating**: → -1 (Ignored in training)

This binary conversion is essential for RBM's probabilistic approach.

## 💡 Dataset Download

If you need to download fresh MovieLens datasets:

1. **MovieLens 1M**: https://grouplens.org/datasets/movielens/1m/
2. **MovieLens 100K**: https://grouplens.org/datasets/movielens/100k/

Extract the zip files to their respective folders in the `data/` directory.

## ⚠️ Important Notes

- **File Encoding**: Use `latin-1` encoding for `.dat` files
- **Separator**: `::` for `.dat` files, `\t` for `.base/.test` files
- **Large Files**: Not committed to git (use .gitignore)
- **Binary Conversion**: Critical for RBM's Bernoulli units

## 📊 Dataset Statistics

### ML-1M:
- **Total Ratings**: 1,000,209
- **Users**: 6,040
- **Movies**: 3,883
- **Sparsity**: ~95.7%
- **Rating Distribution**: 
  - 5 stars: 22.5%
  - 4 stars: 34.2%
  - 3 stars: 27.1%
  - 2 stars: 11.4%
  - 1 star: 4.8%

### ML-100K (u1 split):
- **Training**: 80,000 ratings
- **Test**: 20,000 ratings
- **Users**: 943
- **Movies**: 1,682

Your data setup is **perfect** for RBM training! 🎯