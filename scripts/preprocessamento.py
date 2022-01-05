# ################################################################
# Universidade Federal de Sao Carlos (UFSCAR)
# Aprendizado de Maquina - 2020
# Projeto Final

# Aluno: Vinícius Brandão Crepschi
# RA/CPF: 743601
# ################################################################

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier

def trata_faltantes( df_dataset ):
    # Preenche os valores NaN com a media e retorna o dataset
    df_dataset = df_dataset.apply(lambda x: x.fillna(x.mean()),axis=0)
    
    return df_dataset.round(6)


def normalizar(X):
    # Caso o dataset possua a coluna de classe normaliza sem considerar essa coluna
    if 'Class' in X: 

      X1 = X.iloc[:,0:-1].values
      X_norm = preprocessing.normalize(X1)
      X.iloc[:,0:-1] = X_norm

    # Caso ao contenha (datasets de teste)
    else: 

      X1 = X.iloc[:].values
      X_norm = preprocessing.normalize(X1)
      X.iloc[:] = X_norm

    return X

def feature_selection(X, test):

    x = X.iloc[:, 0:-1].values 

    y = X.iloc[:, -1].values

    # treina um modelo Extra Trees
    model = ExtraTreesClassifier()
    model.fit(x, y)
    # Exibe os indices das colunas de importancia 0
    idx = np.where(model.feature_importances_ == 0)[0]
    print(idx)  

    # Remove da base de treino e teste as colunas selecionadas
    ds_selected = X.drop(X.columns[idx], axis='columns')
    test_selected = test.drop(test.columns[idx], axis='columns')
    
    return ds_selected, test_selected

def build_train_set(ds, train, test):
  
  ds1 = ds.drop(columns=['Id'])
  train_set = ds1.iloc[train['Id'].values].copy()
  train_set['Class'] = train['Class'].values
  train_set.drop(train_set[train_set['Class'] == 0].index, inplace=True)

  train_test = ds1.iloc[test['Id'].values].copy()

  return train_set, train_test