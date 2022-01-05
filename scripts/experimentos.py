# ################################################################
# Universidade Federal de Sao Carlos (UFSCAR)
# Aprendizado de Maquina - 2020
# Projeto Final

# Aluno: Vinícius Brandão Crepschi
# RA/CPF: 743601
# ################################################################

# Arquivo com todas as funcoes e codigos referentes aos experimentos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm 
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np

def train_knn(ds):

  # Pega os valores das n-1 primeiras colunas e guarda em uma matrix X
  X = ds.iloc[:, 0:-1].values

  # Pega os valores da ultima coluna e guarda em um vetor Y
  Y = ds.iloc[:, -1].values
  
  # Inicializa o modelo
  knn = KNeighborsClassifier()

  # Divide a base de treino em treino e teste (80/20)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)

  # Realiza o upsampling
  X_resample, Y_resample = resample(X_train[np.where(Y_train == 1)], Y_train[np.where(Y_train == 1)], n_samples=X_train[np.where(Y_train == -1)].shape[0])
  X_train = np.concatenate((X_train, X_resample), axis=0)
  Y_train = np.concatenate((Y_train, Y_resample), axis=0)

  # Parametros para o grid search
  parameters = {'n_neighbors':[3, 5, 7, 14, 28],
                'weights': ['distance'],
                }

  # Realiza o grid search
  knn = GridSearchCV(knn, param_grid=parameters, verbose=1, n_jobs=-1, scoring='roc_auc')

  # Treina o modelo
  knn.fit(X_train, Y_train)

  # Prediz a base de teste
  Y_pred = knn.predict(X_test)

  return knn, X_test, Y_pred, Y_test

def train_log_reg(ds):
  # Pega os valores das n-1 primeiras colunas e guarda em uma matrix X
  X = ds.iloc[:, 0:-1].values

  # Pega os valores da ultima coluna e guarda em um vetor Y
  Y = ds.iloc[:, -1].values

  # Divide a base de treino em treino e teste (80/20)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)

  # Realiza o upsampling
  X_resample, Y_resample = resample(X_train[np.where(Y_train == 1)], Y_train[np.where(Y_train == 1)], n_samples=X_train[np.where(Y_train == -1)].shape[0])
  X_train = np.concatenate((X_train, X_resample), axis=0)
  Y_train = np.concatenate((Y_train, Y_resample), axis=0)

  # Inicializa o modelo
  logreg = LogisticRegression(verbose=1, max_iter=5000, class_weight='balanced')

  # Parametros para o grid search
  parameters = {'C': [500, 1000, 3000, 5000], 'penalty':['l1','l2'], 'class_weight':['balanced']}

  # Realiza o grid search
  logreg = GridSearchCV(logreg, param_grid = parameters, scoring='roc_auc', n_jobs=-1, cv=5)

  # Treina o modelo
  logreg.fit(X_train, Y_train)

  # Prediz a base de teste
  Y_pred = logreg.predict(X_test)

  return logreg, X_test, Y_pred, Y_test

def train_nb(ds):
      # Pega os valores das n-1 primeiras colunas e guarda em uma matrix X
    X = ds.iloc[:, 0:-1].values

    # Pega os valores da ultima coluna e guarda em um vetor Y
    Y = ds.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)
    X_resample, Y_resample = resample(X_train[np.where(Y_train == 1)], Y_train[np.where(Y_train == 1)], n_samples=round(X_train[np.where(Y_train == -1)].shape[0]))

    X_train = np.concatenate((X_train, X_resample), axis=0)
    Y_train = np.concatenate((Y_train, Y_resample), axis=0)

    nb = GaussianNB()
    nb.fit(X_train, Y_train) 

    # predicting test set results 
    Y_pred = nb.predict(X_test) 

    return nb, X_test, Y_pred, Y_test

def train_svm(ds):

  # Pega os valores das n-1 primeiras colunas e guarda em uma matrix X
  X = ds.iloc[:, 0:-1].values

  # Pega os valores da ultima coluna e guarda em um vetor Y
  Y = ds.iloc[:, -1].values

  # Divide a base de treino em treino e teste (80/20)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)

  # Realiza o upsampling
  X_resample, Y_resample = resample(X_train[np.where(Y_train == 1)], Y_train[np.where(Y_train == 1)], n_samples=X_train[np.where(Y_train == -1)].shape[0])

  X_train = np.concatenate((X_train, X_resample), axis=0)
  Y_train = np.concatenate((Y_train, Y_resample), axis=0)

  # Inicializa o modelo
  supportv = svm.SVC(class_weight='balanced')

  # Parametros para o grid search
  param_grid_svc = [{
                      'C': [0.001, 0.5, 1.0, 10.0, 100, 1000],
                      'kernel': ['linear'],
                      'class_weight':['balanced']
                    },
                  ]

  # Realiza o grid search
  supportv = GridSearchCV(supportv, param_grid=param_grid_svc, verbose=1, n_jobs=-1, scoring='roc_auc')

  # Treina o modelo
  supportv.fit(X_train, Y_train)

  # Prediz a base de teste
  Y_pred =  supportv.predict(X_test)

  return supportv, X_test, Y_pred, Y_test

def train_mlp(ds):

  X = ds.iloc[:, 0:-1].values

  Y = ds.iloc[:, -1].values

  # Divide a base de treino em treino e teste (80/20)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)

  # Realiza o upsampling
  X_resample, Y_resample = resample(X_train[np.where(Y_train == 1)], Y_train[np.where(Y_train == 1)], n_samples=X_train[np.where(Y_train == -1)].shape[0])

  X_train = np.concatenate((X_train, X_resample), axis=0)
  Y_train = np.concatenate((Y_train, Y_resample), axis=0)

  # Parametros para o grid search
  parameters = {
          'solver': ['lbfgs'], 
          'max_iter': [2000], 
          'alpha': 10.0 ** -np.arange(1, 3), 
          'hidden_layer_sizes': [round(ds.shape[1] * (1/3)), round(ds.shape[1] * (2/3)), ds.shape[1]] 
      }

  # Inicializa o modelo
  neural = GridSearchCV(MLPClassifier(verbose=1, max_iter=800), param_grid=parameters, n_jobs=-1, verbose=1, scoring='roc_auc')

  # Treina o modelo
  neural.fit(X_train, Y_train)

  # Prediz a base de teste
  Y_pred =  neural.predict(X_test)

  return neural, X_test, Y_pred, Y_test

def upsample(X, Y):
  X_resample, Y_resample = resample(X[np.where(Y == 1)], Y[np.where(Y == 1)], n_samples=X[np.where(Y == -1)].shape[0])
  X_train = np.concatenate((X, X_resample), axis=0)
  Y_train = np.concatenate((Y, Y_resample), axis=0)

  return X_train, Y_train
