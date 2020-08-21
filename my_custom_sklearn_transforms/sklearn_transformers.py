from sklearn.base import BaseEstimator, TransformerMixin

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

# Custom replace missing values
class ReplaceMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self, mean_columns, zero_columns):
        self.mean_columns = mean_columns
        self.zero_columns = zero_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        
        for mean_column in self.mean_columns:
          data[[mean_column]] = data[[mean_column]].fillna(data[[mean_column]].mean())
        for zero_column in self.zero_columns:
          data[[zero_column]] = data[[zero_column]].fillna(0)

        return data
