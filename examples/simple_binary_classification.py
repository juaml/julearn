from seaborn import load_dataset
from julearn import run_cross_validation

df_iris = load_dataset('iris')

# keep only two species
df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]


X = ['sepal_length', 'sepal_width', 'petal_length']
y = 'species'
scores = run_cross_validation(X=X, y=y, data=df_iris, model='svm')

print(scores)
