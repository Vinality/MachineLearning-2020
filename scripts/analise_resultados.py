# ################################################################
# Universidade Federal de Sao Carlos (UFSCAR)
# Aprendizado de Maquina - 2020
# Projeto Final

# Aluno: Vinícius Brandão Crepschi
# RA/CPF: 743601
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise dos resultados
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np

def plot_roc_auc(Y_test, Y_pred):
  # Plota curva ROC e Area Sob Curva
  y_true = np.where(Y_test == -1, 0, 1)
  fpr, tpr, thresholds = roc_curve(y_true, Y_pred)
  roc_auc = auc(fpr, tpr)

  plt.figure()
  plt.plot(fpr, tpr, label='Curva ROC (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Taxa de Falso Positivo')
  plt.ylabel('Taxa de Verdadeiro Positivo')
  plt.title('Curva ROC')
  plt.legend(loc="lower right")
  plt.show()

def plot_conf_matrix(clf, X_test, Y_test, Y_pred):
  # Imprime o relatorio de metricas e plota matriz de confusao
  print(classification_report(Y_test, Y_pred))
  print(plot_confusion_matrix(clf, X_test, Y_test))

# Seguindo exemplo da documentacao do scipy
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):   

  # Plota graficos de curva de aprendizado e escalabilidade
  if axes is None:
      _, axes = plt.subplots(1, 3, figsize=(20, 5))

  axes[0].set_title(title)
  if ylim is not None:
      axes[0].set_ylim(*ylim)
  axes[0].set_xlabel("Training examples")
  axes[0].set_ylabel("Score")

  train_sizes, train_scores, test_scores, fit_times, _ = \
      learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                      train_sizes=train_sizes,
                      return_times=True)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)
  fit_times_mean = np.mean(fit_times, axis=1)
  fit_times_std = np.std(fit_times, axis=1)

  axes[0].grid()
  axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
  axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
  axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
  axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")
  axes[0].legend(loc="best")

  axes[1].grid()
  axes[1].plot(train_sizes, fit_times_mean, 'o-')
  axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                        fit_times_mean + fit_times_std, alpha=0.1)
  axes[1].set_xlabel("Training examples")
  axes[1].set_ylabel("fit_times")
  axes[1].set_title("Scalability of the model")

  axes[2].grid()
  axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
  axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1)
  axes[2].set_xlabel("fit_times")
  axes[2].set_ylabel("Score")
  axes[2].set_title("Performance of the model")

  return plt