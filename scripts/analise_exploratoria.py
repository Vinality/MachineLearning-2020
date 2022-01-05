# ################################################################
# Universidade Federal de Sao Carlos (UFSCAR)
# Aprendizado de Maquina - 2020
# Projeto Final

# Aluno: Vinícius Brandão Crepschi
# RA/CPF: 743601
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

import numpy as np #importa a biblioteca usada para trabalhar com vetores e matrizes
import pandas as pd #importa a biblioteca usada para trabalhar com dataframes (dados em formato de tabela) e análise de dados
import scipy.stats as stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def encontrar_nan(df_dataset): 
  # Encontra os indices que contem valores NaN 
  idxRowNan = pd.isnull(df_dataset).any(1).to_numpy().nonzero()
  # Exibe as linhas que contem valores NaN
  display(df_dataset.iloc[idxRowNan])

def plot_pca(X):
  x = X.iloc[:, 0:-1]

  # Realiza a trasnformacao para duas dimensoes
  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(x)

  # Plota as duas dimensoes do PCA
  plt.scatter(X_pca[:,0], X_pca[:,1])
  plt.show

def plot_distribuicao(X):
  # Conta e exibe a distribuicao das classes
  sns.countplot(x="Class", data=X)

  plt.show()
  display(X['Class'].value_counts().sort_index())

def boxplot_ds(X):
  # Plota boxplot do dataset
  X.boxplot(figsize=(15,7))
  plt.show()

