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

        mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        zero_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

        for mean_column in self.mean_columns:
          data[[mean_column]] = mean_imputer.fit_transform(data[[mean_column]])
        for zero_column in self.zero_columns:
          data[[zero_column]] = zero_imputer.fit_transform(data[[zero_column]])

        return data
