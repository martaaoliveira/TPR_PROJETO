from pyclbr import Class
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn import svm
import seaborn as sns
import time
import sys
import warnings
import pandas as pd
warnings.filterwarnings('ignore')


def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")

## -- 3 -- ##

# features: É a matriz de características (ou atributos) onde cada linha representa uma observação e cada coluna representa um atributo específico.
# oClass: É a matriz de classes correspondentes a cada observação features.


def plotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    #blue Marta
    #green Bruno
    #RED for Dns tunneling

    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    # Adicionar nomes aos eixos e título
    plt.xlabel(f'Feature {f1index}')
    plt.ylabel(f'Feature {f2index}')
    plt.title(f'Gráfico de Features {f1index} vs {f2index}')

    plt.show()
    waitforEnter()

def logplotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.loglog(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()

## -- 11 -- ##
def distance(c,p):
    s=0
    n=0
    for i in range(len(c)):
        if c[i]>0:
            s+=np.square((p[i]-c[i])/c[i])
            n+=1

    return(np.sqrt(s/n))

    #return(np.sqrt(np.sum(np.square((p-c)/c))))

########### Main Code #############


Classes={0:'Bruno',1:'Marta',2:'DNS'}
plt.ion()
nfig=1

## -- 2 -- ##
features_bruno=np.loadtxt("features_bruno.dat")
features_marta=np.loadtxt("features_marta.dat")
features_dns=np.loadtxt("features_dns_tunneling.dat")

#It assigns class labels (0 for Bruno, 1 for Marta, and 2 for dns_tunneling) to the respective datasets
#cada classe vai conter:mean, median and standard deviation  and also the silence periods features(mean median and deviation) and percentis for upload and download 
oClass_bruno=np.ones((len(features_bruno),1))*0
oClass_marta=np.ones((len(features_marta),1))*1
oClass_dns=np.ones((len(features_dns),1))*2


#resulta num conjunto de features que contém todos os dados dessas diferentes fontes combinados verticalmente.
features=np.vstack((features_marta,features_bruno,features_dns))
#um único array oClass que contém todas as classes correspondentes às observações do conjunto de dados combinado features.
oClass=np.vstack((oClass_marta,oClass_bruno,oClass_dns))

# scaler = MaxAbsScaler().fit(features)
# features_scaled=scaler.transform(features)


#print(oClass)

# nPktUp BytesUp nPktDown BytesDown
#0..10   11..21  22..32   33..43


#nPktUp
#MAXIMO MEDIA MEDIANA DESVIO SI SI SI PER PER PER PER
# 0      1      2      3      4  5  6  7   8   9   10

#BytesUp
#MAXIMO MEDIA MEDIANA DESVIO SI SI SI PER PER PER PER
# 11      12     13      14  15 16 17 18  19   20  21

#nPktDown
#MAXIMO MEDIA MEDIANA DESVIO SI SI SI PER PER PER PER
# 22      23     24      25  26 27 28 29  30   31  32

#BytesDown
#MAXIMO MEDIA MEDIANA DESVIO SI SI SI PER PER PER PER
# 33      34     35      36  37 38 39 40  41   42 43


#Para a primeira captura do script dumb de dns tunneling os valores de download sao muito mais elevados do que o comportamento normal -> exfiltração de dados 
#script mais inteligente->mostra que o maximo de tráfego são valores mais baixos e estao na mesmo intervalo de valores do comportamento normal -> ataque mais súbtil

# plt.figure(1)
# plt.scatter(features_bruno[:, 0], features_bruno[:, 22], c='blue', label='Bruno')
# plt.scatter(features_marta[:, 0], features_marta[:, 22], c='green', label='Marta')
# plt.scatter(features_dns[:, 0], features_dns[:, 22], c='red', label='DNS Tunneling')
# plt.title('Comparação entre maximo  nPktUp vs maximo  nPktDown')
# plt.xlabel('maximo  nPktUp')  # Rótulo do eixo x
# plt.ylabel('maximo  nPktDown')  # Rótulo do eixo y
# plt.legend(loc='lower right', title='Classes')


#----------------------------------------------------------------------------------------------------------------
#Comparação entre média nPktUpload e desvio padrão nPktUpload
#quantidade de pacotes de upload varia consideravelmente entre diferentes pontos de medição, o que se reflete no aumento do desvio padrão. Existe uma grande quantidade de 
#uploads anormal relativamente ao bom comportamento
#dns_tunneling_smart -> média de pacotes do dns_tunneling é muito inferior ao outro script -> mais subtil
#mesmo assim desvio padrao do ataque é maior que o comportamento normal
# plt.figure(2)
# plt.scatter(features_bruno[:, 1], features_bruno[:, 3], c='blue', label='Bruno')
# plt.scatter(features_marta[:, 1], features_marta[:, 3], c='green', label='Marta')
# plt.scatter(features_dns[:, 1], features_dns[:, 3], c='red', label='DNS Tunneling')
# plt.title('Comparação entre média nPktUp e desvio padrão nPktUp')
# plt.xlabel('média nPktUP')  # Rótulo do eixo x
# plt.ylabel('desvio padrão nPktUp')  # Rótulo do eixo y
# plt.legend(loc='lower right', title='Classes')

#------------------------------------------------------

#Comparar media silencio nPktUp com media desvio padrao nPktUp
#upload -> exfiltração de dados
#No caso do bom comportamento existe claramente um comportamento humano pois não existe uma linearidade de valores, resultados mais espaçados, maior numero de silencios -> explicar que
#cada um fez um tipo de browsing e dessa forma tem um comportamento diferente (eu tinha add block)
#para o caso do dns_tunneling_burro não existe tanta diferença entre a media e o desvio padrao e existe menos silencio -> extração de dados sem limite
#Desvio padrao pequeno (e quase constante) -> comportamento de um bot (nao humano) -> script mongloide


#script_maior -> mais silencios relativamente ao script burro. Ainda assim o desvio padrao pequeno o que indica ainda assim comportamento anormal. O desvio do gaussian delay foi 
#  talvez demasiado pequeno entre transferencia de chunks -> mesmo com gaussian delay nao é facil imitar comportamento humano
# plt.figure(3)
# plt.scatter(features_bruno[:, 5], features_bruno[:, 6], c='blue', label='Bruno')
# plt.scatter(features_marta[:, 5], features_marta[:, 6], c='green', label='Marta')
# plt.scatter(features_dns[:, 5], features_dns[:, 6], c='red', label='DNS Tunneling')
# plt.title('Comparação entre média de silêncio nPktUp e desvio padrão silêncio nPktUp')
# plt.xlabel('média silêncio nPktUp')  # Rótulo do eixo x
# plt.ylabel('desvio padrão silêncio nPktUp')  # Rótulo do eixo y
# plt.legend(loc='lower right', title='Classes')


#------------------------------------------------------
#Comparação entre desvio BytesUp e percentis 98 BytesUp
# Dispersão dos Dados: Um desvio padrão maior indica maior variabilidade nos dados. 
# Se o desvio padrão for alto e o percentil 98 também for alto, isso sugere que os dados possuem uma ampla dispersão em torno da média e que há uma presença 
# significativa de valores extremos.
# Outliers: Se o percentil 98 for muito maior do que o desvio padrão, isso pode indicar a presença de outliers, ou seja, valores extremamente altos que estão distantes da média. 
# Isso pode ser significativo para entender situações em que ocorrem transferências de grandes volumes de dados em comparação com a maioria dos casos.
# Estabilidade dos Dados: Um desvio padrão pequeno em relação ao percentil 98 sugere uma menor variabilidade nos dados, 
# indicando maior estabilidade nos volumes de dados transferidos na maioria dos casos.

#Pouca variação no comportamento humanos -> poucos outliers 
#Na exfiltração de dados há imensos outliers -> percentil 98 muito maior que o desvio padrão -> valores distantes da média -> transferencia de grande volumes de dados em 
#comparação com a média


#script smart-> apesar de melhorias(valores de percentis mais baixos ) mesmo assim o script revela uma diferença significativa entre percentis e desvio padrão o que indica 
#uma quantidade de trafego consideravel a ser transmitdo relativamente ao comportamento normal -> exfiltração de dados
# plt.figure(4)
# plt.scatter(features_bruno[:, 14], features_bruno[:, 21], c='blue', label='Bruno')
# plt.scatter(features_marta[:, 14], features_marta[:, 21], c='green', label='Marta')
# plt.scatter(features_dns[:, 14], features_dns[:, 21], c='red', label='DNS Tunneling')
# plt.title('Comparação entre desvio BytesUp e percentis 98 BytesUp')
# plt.xlabel('desvio BytesUp ')  # Rótulo do eixo x
# plt.ylabel('percentis 98 BytesUp')  # Rótulo do eixo y
# plt.legend(loc='lower right', title='Classes')


#divisão do conjunto de dados em dados de treino e de teste.

## -- 3 -- ##
#:i
#Define a percentagem dos dados originais que serão usados para TREINO (50% neste caso).
percentage=0.5
#pB, pYT, pM: Calculam o tamanho do conjunto de treino para cada categoria com base na percentagem definida.
pB=int(len(features_bruno)*percentage)
trainFeatures_bruno=features_bruno[:pB,:]
pM=int(len(features_marta)*percentage)
trainFeatures_marta=features_marta[:pM,:]
pD=int(len(features_dns)*percentage)
trainFeatures_dns=features_dns[:pD,:]

#i2train: Build train features of normal behavior
# i2train=np.vstack((trainFeatures_bruno,trainFeatures_marta))
# o2trainClass=np.vstack((oClass_bruno[:pB],oClass_marta[:pM]))
i2train=np.vstack((trainFeatures_bruno))
o2trainClass=np.vstack((oClass_bruno[:pB]))

#:ii
# i3Ctrain=np.vstack((trainFeatures_bruno,trainFeatures_marta,trainFeatures_dns))
# o3trainClass=np.vstack((oClass_bruno[:pB],oClass_marta[:pM],oClass_dns[:pD]))
i3Ctrain=np.vstack((trainFeatures_bruno,trainFeatures_dns))
o3trainClass=np.vstack((oClass_bruno[:pB],oClass_dns[:pD]))

#:iii
testFeatures_bruno=features_bruno[pB:,:]
testFeatures_marta=features_marta[pM:,:]
testFeatures_dns=features_dns[pD:,:]
#Fornecer ao modelos os dados de TESTE
i3Atest=np.vstack((testFeatures_bruno,testFeatures_dns))
o3testClass=np.vstack((oClass_bruno[pB:],oClass_dns[pD:]))

############################################## -- 7 -- Centroids Distances SEM PCA ######################### 

# # i2train=np.vstack((trainFeatures_bruno,trainFeatures_marta))
# i2train=np.vstack((trainFeatures_bruno))
# #scaler = MaxAbsScaler().fit(i2train)
# #i2train=scaler.transform(i2train)

# results = []
# actual_labels = []
# predicted_labels = []
# centroids={}
# best_f1_index = 0  # Initializing the index of the best F1 score

# for c in range(1):  # Only the first  class(client class)
#     pClass=(o2trainClass==c).flatten()
#     centroids.update({c:np.mean(i2train[pClass,:],axis=0)})
# #print('All Features Centroids:\n',centroids)

# # i3Atest=np.vstack((testFeatures_bruno,testFeatures_marta,testFeatures_dns))
# i3Atest=np.vstack((testFeatures_bruno,testFeatures_dns))
# #i3Atest=scaler.transform(i3Atest)

# print('\n-- Anomaly Detection based on Centroids Distances without PCA --')
# nObsTest,nFea=i3Atest.shape

# # Define a range of threshold values to test
# threshold_values = [0.1,0.2,0.3,0.4,0.5,1.0,1.2, 1.1, 1.5, 2.0,3,5,6,10] 

# # Initialize lists to store evaluation metrics for each threshold
# threshold_metrics = []


# best_confusion_matrix = None  # Initializing outside the conditional block

# for AnomalyThreshold in threshold_values:
#     # Initialize variables for TP, FP, TN, FN for the current threshold
#     tp_centroids = 0
#     fp_centroids = 0
#     tn_centroids = 0
#     fn_centroids = 0
#     actual_labels = []
#     predicted_labels = []

#     # Perform anomaly detection based on the current threshold
#     for i in range(nObsTest):
#         actual_labels.append(o3testClass[i][0])
#         x = i3Atest[i]
#         dists = [distance(x, centroids[0])]
#         if min(dists) > AnomalyThreshold:
#             result = "Anomaly"
#             predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
#             if o3testClass[i][0] == 2.0:  # DNS class
#                 tp_centroids += 1  # True Positive
#             else:
#                 fp_centroids += 1  # False Positive
#         else:
#             result = "OK"
#             predicted_labels.append(0.0) 
#             if o3testClass[i][0] == 2.0:  # DNS class
#                 fn_centroids += 1  # False Negative
#             else:
#                 tn_centroids += 1  # True Negative


#     # Calculate evaluation metrics for the current threshold
#     accuracy_centroids = ((tp_centroids + tn_centroids) / nObsTest) * 100
#     precision_centroids = (tp_centroids / (tp_centroids + fp_centroids)) * 100 if (tp_centroids + fp_centroids) != 0 else 0
#     recall_centroids = (tp_centroids / (tp_centroids + fn_centroids)) * 100 if (tp_centroids + fn_centroids) != 0 else 0
#     f1_score_centroids = (2 * (precision_centroids * recall_centroids) / (precision_centroids + recall_centroids)) if (precision_centroids + recall_centroids) != 0 else 0

#     # Store metrics for the current threshold in the list
#     results.append({
#         'AnomalyThreshold': AnomalyThreshold,
#         'TP': tp_centroids,
#         'FP': fp_centroids,
#         'TN': tn_centroids,
#         'FN': fn_centroids,
#         'Accuracy': accuracy_centroids,
#         'Precision': precision_centroids,
#         'Recall': recall_centroids,
#         'F1 Score': f1_score_centroids,
#         'ConfusionMatrix': confusion_matrix(actual_labels, predicted_labels)
#     })


# df = pd.DataFrame(results)

# df.to_excel('resultados_centroid.xlsx', index=False)

# # Find the index of the row with the best F1 score
# best_f1_index = df['F1 Score'].idxmax()

# # Get the threshold with the best F1 score
# best_threshold = df.loc[best_f1_index, 'AnomalyThreshold']

# best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']

# # Plot the best confusion matrix if it exists
# plt.figure(figsize=(8, 6))
# sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
#                 xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Best Confusion Matrix (Threshold: {best_threshold})')
# plt.show()


# ####################################### -- 7.2 -- Centroids Distances Com PCA ######################### ##

# ##print('\n-- Anomaly Detection based on Centroids Distances with PCA --')
# components_to_test = [10, 15, 20, 25]  
# results = []
# best_confusion_matrix = None  # Initializing outside the conditional block

# for n_components in components_to_test:
#     i2train = np.vstack((trainFeatures_bruno))

#     actual_labels = []
#     predicted_labels = []

#     # Initialize PCA and fit-transform the data
#     pca = PCA(n_components=n_components)
#     i2train_pca = pca.fit_transform(i2train)

#     centroids = {}
#     for c in range(1):  
#         pClass = (o2trainClass == c).flatten()
#         centroids.update({c: np.mean(i2train_pca[pClass, :], axis=0)})

#     i3Atest = np.vstack((testFeatures_bruno, testFeatures_dns))
#     i3Atest_pca = pca.transform(i3Atest)

#     print(f'\n-- Anomaly Detection based on Centroids Distances (Components: {n_components}) --')
#     nObsTest, nFea = i3Atest_pca.shape

#     # Define a range of threshold values to test
#     threshold_values = [0.1,0.2,0.3,0.4,0.5,1.0,1.2, 1.1, 1.5, 2.0,3,5,6,10]  # Add more threshold values as needed

#     threshold_metrics = []

#     for AnomalyThreshold in threshold_values:
#         # Initialize variables for TP, FP, TN, FN for the current threshold
#         tp_centroids = 0
#         fp_centroids = 0
#         tn_centroids = 0
#         fn_centroids = 0
#         actual_labels = []
#         predicted_labels = []

#         # Perform anomaly detection based on the current threshold
#         for i in range(nObsTest):
#             actual_labels.append(o3testClass[i][0])
#             x = i3Atest_pca[i]
#             dists = [distance(x, centroids[0])]
#             if min(dists) > AnomalyThreshold:
#                 result = "Anomaly"
#                 predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
#                 if o3testClass[i][0] == 2.0:  # DNS class
#                     tp_centroids += 1  # True Positive
#                 else:
#                     fp_centroids += 1  # False Positive

#             else:
#                 result = "OK"
#                 predicted_labels.append(0.0) 
#                 if o3testClass[i][0] == 2.0:  # DNS class
#                     fn_centroids += 1  # False Negative
#                 else:
#                     tn_centroids += 1  # True Negative

#         # Calculate evaluation metrics for the current threshold
#         accuracy_centroids = ((tp_centroids + tn_centroids) / nObsTest) * 100
#         precision_centroids = (tp_centroids / (tp_centroids + fp_centroids)) * 100 if (tp_centroids + fp_centroids) != 0 else 0
#         recall_centroids = (tp_centroids / (tp_centroids + fn_centroids)) * 100 if (tp_centroids + fn_centroids) != 0 else 0
#         f1_score_centroids = (2 * (precision_centroids * recall_centroids) / (precision_centroids + recall_centroids))  if (precision_centroids + recall_centroids) != 0 else 0

#         # Store metrics for the current threshold in the list
#         results.append({
#             'AnomalyThreshold': AnomalyThreshold,
#             'Número de Componentes': n_components,  
#             'TP': tp_centroids,
#             'FP': fp_centroids,
#             'TN': tn_centroids,
#             'FN': fn_centroids,
#             'Accuracy': accuracy_centroids,
#             'Precision': precision_centroids,
#             'Recall': recall_centroids,
#             'F1 Score': f1_score_centroids,
#             'ConfusionMatrix': confusion_matrix(actual_labels, predicted_labels)

#         })


# df = pd.DataFrame(results)
# df.to_excel('resultados_centroid_pca.xlsx', index=False)

# best_f1_index = df['F1 Score'].idxmax()
# best_threshold = df.loc[best_f1_index, 'AnomalyThreshold']
# best_number_components=df.loc[best_f1_index,'Número de Componentes']
# best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']

# # Plot the best confusion matrix if it exists
# plt.figure(figsize=(8, 6))
# sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Best Confusion Matrix: Threshold: {best_threshold} Number of components: {best_number_components}')
# plt.show()



########################################## -- 8 -- Anomaly Detection based on One Class Support Vector Machines WITHOUT PCA ###############################


# tp_linear, fn_linear, tn_linear, fp_linear = 0, 0, 0, 0
# tp_rbf, fn_rbf, tn_rbf, fp_rbf = 0, 0, 0, 0
# tp_poly, fn_poly, tn_poly, fp_poly = 0, 0, 0, 0
# i2train=np.vstack((trainFeatures_bruno))
# i3Atest=np.vstack((trainFeatures_bruno,testFeatures_dns))
# results=[]
# AnomResults={-1:"Anomaly",1:"OK"}
# nObsTest,nFea=i3Atest.shape

# nu=0.5
# ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear',nu=nu).fit(i2train)
# rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf',nu=nu).fit(i2train)
# poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',nu=nu,degree=2).fit(i2train)

# L1=ocsvm.predict(i3Atest)
# L2=rbf_ocsvm.predict(i3Atest)
# L3=poly_ocsvm.predict(i3Atest)

# actual_labels_linear = []
# predicted_labels_linear = []

# actual_labels_rbf = []
# predicted_labels_rbf = []

# actual_labels_poly = []
# predicted_labels_poly = []


# for i in range(nObsTest):
#     #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
#         actual_labels_linear.append(o3testClass[i][0])
#         actual_labels_rbf.append(o3testClass[i][0])
#         actual_labels_poly.append(o3testClass[i][0])

#         # Linear
#         if AnomResults[L1[i]] == "Anomaly":
#             predicted_labels_linear.append(2.0)  # Predicted as DNS (anomaly)
#             if o3testClass[i][0] == 2:  # DNS class
#                 tp_linear += 1 # True Positive
#             else:  # Marta/Bruno class
#                 fp_linear += 1 # False Positive
#         else: # OK
#             predicted_labels_linear.append(0.0) 
#             if o3testClass[i][0] == 2:
#                 fn_linear += 1  # False Negative
#             else:
#                 tn_linear += 1  # True Negative

#             # RBF
#         if AnomResults[L2[i]] == "Anomaly":
#             predicted_labels_rbf.append(2.0)  # Predicted as DNS (anomaly)
#             if o3testClass[i][0] == 2:  # DNS class
#                 tp_rbf += 1 # True Positive
#             else:  # Marta/Bruno class
#                 fp_rbf += 1 # False Positive
#         else: # OK
#             predicted_labels_rbf.append(0.0) 
#             if o3testClass[i][0] == 2:
#                 fn_rbf += 1  # False Negative
#             else:
#                 tn_rbf += 1  # True Negative
#             # Poly
#         if AnomResults[L3[i]] == "Anomaly":
#             predicted_labels_poly.append(2.0)  # Predicted as DNS (anomaly)
#             if o3testClass[i][0] == 2:  # DNS class
#                 tp_poly += 1 # True Positive
#             else:  # Marta/Bruno class
#                 fp_poly += 1 # False Positive
#         else: # OK
#             predicted_labels_poly.append(0.0) 
#             if o3testClass[i][0] == 2:
#                 fn_poly += 1  # False Negative
#             else:
#                 tn_poly += 1  # True Negative
# accuracy_linear = ((tp_linear + tn_linear) / nObsTest) * 100
# precision_linear = (tp_linear / (tp_linear + fp_linear)) * 100 if tp_linear + fp_linear > 0 else 0
# recall_linear = (tp_linear / (tp_linear + fn_linear)) * 100 if tp_linear + fn_linear > 0 else 0
# f1_score_linear = (2 * (precision_linear * recall_linear) / (precision_linear + recall_linear)) / 100 if (precision_linear + recall_linear) != 0 else 0

# accuracy_rbf = ((tp_rbf + tn_rbf) / nObsTest) * 100
# precision_rbf = (tp_rbf / (tp_rbf + fp_rbf)) * 100 if tp_rbf + fp_rbf > 0 else 0
# recall_rbf = (tp_rbf / (tp_rbf + fn_rbf)) * 100 if tp_rbf + fn_rbf > 0 else 0
# f1_score_rbf = (2 * (precision_rbf * recall_rbf) / (precision_rbf + recall_rbf)) / 100 if (precision_rbf + recall_rbf) != 0 else 0

# accuracy_poly = ((tp_poly + tn_poly) / nObsTest) * 100
# precision_poly = (tp_poly / (tp_poly + fp_poly)) * 100 if tp_poly + fp_poly > 0 else 0
# recall_poly = (tp_poly / (tp_poly + fn_poly)) * 100 if tp_poly + fn_poly > 0 else 0
# f1_score_poly = (2 * (precision_poly * recall_poly) / (precision_poly + recall_poly)) / 100 if (precision_poly + recall_poly) != 0 else 0


# results = {
#     'Method': ['Linear', 'RBF', 'Poly'],
#     'TP': [tp_linear, tp_rbf, tp_poly],
#     'FP': [fp_linear, fp_rbf, fp_poly],
#     'TN': [tn_linear, tn_rbf, tn_poly],
#     'FN': [fn_linear, fn_rbf, fn_poly],
#     'Accuracy': [accuracy_linear, accuracy_rbf, accuracy_poly],
#     'Precision': [precision_linear, precision_rbf, precision_poly],
#     'Recall': [recall_linear, recall_rbf, recall_poly],
#     'F1 Score': [f1_score_linear, f1_score_rbf, f1_score_poly],
#     'ConfusionMatrix': [confusion_matrix(actual_labels_linear, predicted_labels_linear),confusion_matrix(actual_labels_rbf, predicted_labels_rbf),confusion_matrix(actual_labels_poly, predicted_labels_poly)]

# }
# # # Crie um DataFrame a partir da lista de resultados
# df = pd.DataFrame(results)

# # # Salve o DataFrame em um arquivo Excel
# df.to_excel('resultados_OneClassSVM.xlsx', index=False)

# # Find the index of the row with the best F1 score
# best_f1_index = df['F1 Score'].idxmax()

# best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']
# best_method =df.loc[best_f1_index, 'Method']

# # Plot the best confusion matrix if it exists
# plt.figure(figsize=(8, 6))
# sns.heatmap(best_confusion_matrix, annot=True, cmap='Oranges', fmt='d',
#                 xticklabels=['Negative', 'Positive'], yticklabels=['False', 'True'])
# plt.xlabel('Predicted label')
# plt.ylabel('Actual label')
# plt.title(f'Best Confusion Matrix (Kernel:{best_method} )')
# plt.show()



# ##################################################################################### -- 8.2 -- Anomaly Detection based on One Class Support Vector Machines with pca###############################

# n_components_list = [1, 5, 10, 15, 16, 17, 18, 19, 20, 21]

# results = []
# all_results = []

# # Loop pelos diferentes números de componentes PCA
# for n_components in n_components_list:
#     # Reduzindo a dimensionalidade com PCA
#     pca = PCA(n_components=n_components)
#     i2train_pca = pca.fit_transform(i2train)
#     i3Atest_pca = pca.transform(i3Atest)

#     nu = 0.5
#     ocsvm = OneClassSVM(gamma='scale', kernel='linear', nu=nu).fit(i2train_pca)
#     rbf_ocsvm = OneClassSVM(gamma='scale', kernel='rbf', nu=nu).fit(i2train_pca)
#     poly_ocsvm = OneClassSVM(gamma='scale', kernel='poly', nu=nu, degree=2).fit(i2train_pca)

#     L1 = ocsvm.predict(i3Atest_pca)
#     L2 = rbf_ocsvm.predict(i3Atest_pca)
#     L3 = poly_ocsvm.predict(i3Atest_pca)
    
#     tp_linear, fn_linear, tn_linear, fp_linear = 0, 0, 0, 0
#     tp_rbf, fn_rbf, tn_rbf, fp_rbf = 0, 0, 0, 0
#     tp_poly, fn_poly, tn_poly, fp_poly = 0, 0, 0, 0
    
#     AnomResults={-1:"Anomaly",1:"OK"}
#     actual_labels_linear = []
#     predicted_labels_linear = []

#     actual_labels_rbf = []
#     predicted_labels_rbf = []

#     actual_labels_poly = []
#     predicted_labels_poly = []

#     nObsTest,nFea=i3Atest_pca.shape
#     for i in range(nObsTest):
#         actual_labels_linear.append(o3testClass[i][0])
#         actual_labels_rbf.append(o3testClass[i][0])
#         actual_labels_poly.append(o3testClass[i][0])

#         # Linear
#         if AnomResults[L1[i]] == "Anomaly":
#             predicted_labels_linear.append(2.0)  # Predicted as DNS (anomaly)
#             if o3testClass[i][0] == 2:  # DNS class
#                 tp_linear += 1 # True Positive
#             else:  # Marta/Bruno class
#                 fp_linear += 1 # False Positive
#         else: # OK
#             predicted_labels_linear.append(0.0) 
#             if o3testClass[i][0] == 2:
#                 fn_linear += 1  # False Negative
#             else:
#                 tn_linear += 1  # True Negative

#             # RBF
#         if AnomResults[L2[i]] == "Anomaly":
#             predicted_labels_rbf.append(2.0)  # Predicted as DNS (anomaly)
#             if o3testClass[i][0] == 2:  # DNS class
#                 tp_rbf += 1 # True Positive
#             else:  # Marta/Bruno class
#                 fp_rbf += 1 # False Positive
#         else: # OK
#             predicted_labels_rbf.append(0.0) 
#             if o3testClass[i][0] == 2:
#                 fn_rbf += 1  # False Negative
#             else:
#                 tn_rbf += 1  # True Negative
#             # Poly
#         if AnomResults[L3[i]] == "Anomaly":
#             predicted_labels_poly.append(2.0)  # Predicted as DNS (anomaly)
#             if o3testClass[i][0] == 2:  # DNS class
#                 tp_poly += 1 # True Positive
#             else:  # Marta/Bruno class
#                 fp_poly += 1 # False Positive
#         else: # OK
#             predicted_labels_poly.append(0.0) 
#             if o3testClass[i][0] == 2:
#                 fn_poly += 1  # False Negative
#             else:
#                 tn_poly += 1  # True Negative

    
#     accuracy_linear = ((tp_linear + tn_linear) / nObsTest) * 100
#     precision_linear = (tp_linear / (tp_linear + fp_linear)) * 100 if tp_linear + fp_linear > 0 else 0

#     accuracy_rbf = ((tp_rbf + tn_rbf) / nObsTest) * 100
#     precision_rbf = (tp_rbf / (tp_rbf + fp_rbf)) * 100 if tp_rbf + fp_rbf > 0 else 0

#     accuracy_poly = ((tp_poly + tn_poly) / nObsTest) * 100
#     precision_poly = (tp_poly / (tp_poly + fp_poly)) * 100 if tp_poly + fp_poly > 0 else 0

#     recall_linear = (tp_linear / (tp_linear + fn_linear)) * 100 if tp_linear + fn_linear > 0 else 0
#     recall_rbf = (tp_rbf / (tp_rbf + fn_rbf)) * 100 if tp_rbf + fn_rbf > 0 else 0
#     recall_poly = (tp_poly / (tp_poly + fn_poly)) * 100 if tp_poly + fn_poly > 0 else 0

#     f1_score_linear = (2 * (precision_linear * recall_linear) / (precision_linear + recall_linear)) / 100 if (precision_linear + recall_linear) != 0 else 0
#     f1_score_rbf = (2 * (precision_rbf * recall_rbf) / (precision_rbf + recall_rbf)) / 100 if (precision_rbf + recall_rbf) != 0 else 0
#     f1_score_poly = (2 * (precision_poly * recall_poly) / (precision_poly + recall_poly)) / 100 if (precision_poly + recall_poly) != 0 else 0
    
#     results = {
#         'Method': ['Linear', 'RBF', 'Poly'],
#         'Number components':n_components,
#         'TP': [tp_linear, tp_rbf, tp_poly],
#         'FP': [fp_linear, fp_rbf, fp_poly],
#         'TN': [tn_linear, tn_rbf, tn_poly],
#         'FN': [fn_linear, fn_rbf, fn_poly],
#         'Accuracy': [accuracy_linear, accuracy_rbf, accuracy_poly],
#         'Precision': [precision_linear, precision_rbf, precision_poly],
#         'Recall': [recall_linear, recall_rbf, recall_poly],
#         'F1 Score': [f1_score_linear, f1_score_rbf, f1_score_poly],
#         'ConfusionMatrix': [confusion_matrix(actual_labels_linear, predicted_labels_linear),confusion_matrix(actual_labels_rbf, predicted_labels_rbf),confusion_matrix(actual_labels_poly, predicted_labels_poly)]

#     }
#     all_results.append(results)

# # DataFrame a partir dos resultados
# df = pd.concat([pd.DataFrame(res) for res in all_results], ignore_index=True)

# #DataFrame num  arquivo Excel
# df.to_excel('resultados_OneClassseSvm_pca.xlsx', index=False)

# # Find the index of the row with the best F1 score
# best_f1_index = df['F1 Score'].idxmax()

# best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']

# # Plot the best confusion matrix if it exists
# plt.figure(figsize=(8, 6))
# sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
#                 xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Best Confusion Matrix (Components: {n_components})')
# plt.show()

# ################################################################## -- 10 Classification based on Support Vector Machines without PCA -- #####################################################################################
# print('\n-- Classification based on Support Vector Machines --')

# i3train=np.vstack((trainFeatures_bruno,trainFeatures_dns))
# i3Ctest=np.vstack((testFeatures_bruno,testFeatures_dns))


# svc = svm.SVC(kernel='linear').fit(i3train, o3trainClass)
# rbf_svc = svm.SVC(kernel='rbf').fit(i3train, o3trainClass)
# poly_svc = svm.SVC(kernel='poly',degree=2).fit(i3train, o3trainClass)

# L1=svc.predict(i3Ctest)
# L2=rbf_svc.predict(i3Ctest)
# L3=poly_svc.predict(i3Ctest)


# tp_linear, fn_linear, tn_linear, fp_linear = 0, 0, 0, 0
# actual_labels_linear = []
# predicted_labels_linear = []


# tp_rbf, fn_rbf, tn_rbf, fp_rbf = 0, 0, 0, 0
# actual_labels_rbf = []
# predicted_labels_rbf = []

# tp_poly, fn_poly, tn_poly, fp_poly = 0, 0, 0, 0
# actual_labels_poly = []
# predicted_labels_poly = []

# nObsTest,nFea=i3Ctest.shape

# AnomResults={2.0:"Anomaly",0:"OK"} #Bruno é 0 e DNS é 2

# for i in range(nObsTest):
#     #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],Classes[L1[i]],Classes[L2[i]],Classes[L3[i]]))
#     actual_labels_linear.append(o3testClass[i][0])
#     actual_labels_rbf.append(o3testClass[i][0])
#     actual_labels_poly.append(o3testClass[i][0])
#     # Linear
#     if AnomResults[L1[i]] == "Anomaly":
#         predicted_labels_linear.append(2.0)  # Predicted as DNS (anomaly)
#         if o3testClass[i][0] == 2:  # DNS class
#             tp_linear += 1 # True Positive
#         else:  # Marta/Bruno class
#             fp_linear += 1 # False Positive
#     else: # OK
#         predicted_labels_linear.append(0.0) 
#         if o3testClass[i][0] == 2:
#             fn_linear += 1  # False Negative
#         else:
#             tn_linear += 1  # True Negative

#         # RBF
#     if AnomResults[L2[i]] == "Anomaly":
#         predicted_labels_rbf.append(2.0)  # Predicted as DNS (anomaly)
#         if o3testClass[i][0] == 2:  # DNS class
#             tp_rbf += 1 # True Positive
#         else:  # Marta/Bruno class
#             fp_rbf += 1 # False Positive
#     else: # OK
#         predicted_labels_rbf.append(0.0) 
#         if o3testClass[i][0] == 2:
#             fn_rbf += 1  # False Negative
#         else:
#             tn_rbf += 1  # True Negative
#         # Poly
#     if AnomResults[L3[i]] == "Anomaly":
#         predicted_labels_poly.append(2.0)  # Predicted as DNS (anomaly)
#         if o3testClass[i][0] == 2:  # DNS class
#             tp_poly += 1 # True Positive
#         else:  # Marta/Bruno class
#             fp_poly += 1 # False Positive
#     else: # OK
#         predicted_labels_poly.append(0.0) 
#         if o3testClass[i][0] == 2:
#             fn_poly += 1  # False Negative
#         else:
#             tn_poly += 1  # True Negative


    
# accuracy_linear = ((tp_linear + tn_linear) / nObsTest) * 100
# precision_linear = (tp_linear / (tp_linear + fp_linear)) * 100 if tp_linear + fp_linear > 0 else 0

# accuracy_rbf = ((tp_rbf + tn_rbf) / nObsTest) * 100
# precision_rbf = (tp_rbf / (tp_rbf + fp_rbf)) * 100 if tp_rbf + fp_rbf > 0 else 0

# accuracy_poly = ((tp_poly + tn_poly) / nObsTest) * 100
# precision_poly = (tp_poly / (tp_poly + fp_poly)) * 100 if tp_poly + fp_poly > 0 else 0

# recall_linear = (tp_linear / (tp_linear + fn_linear)) * 100 if tp_linear + fn_linear > 0 else 0
# recall_rbf = (tp_rbf / (tp_rbf + fn_rbf)) * 100 if tp_rbf + fn_rbf > 0 else 0
# recall_poly = (tp_poly / (tp_poly + fn_poly)) * 100 if tp_poly + fn_poly > 0 else 0

# f1_score_linear = (2 * (precision_linear * recall_linear) / (precision_linear + recall_linear)) / 100 if (precision_linear + recall_linear) != 0 else 0
# f1_score_rbf = (2 * (precision_rbf * recall_rbf) / (precision_rbf + recall_rbf)) / 100 if (precision_rbf + recall_rbf) != 0 else 0
# f1_score_poly = (2 * (precision_poly * recall_poly) / (precision_poly + recall_poly)) / 100 if (precision_poly + recall_poly) != 0 else 0

# results = {
#     'Method': ['Linear', 'RBF', 'Poly'],
#     'TP': [tp_linear, tp_rbf, tp_poly],
#     'FP': [fp_linear, fp_rbf, fp_poly],
#     'TN': [tn_linear, tn_rbf, tn_poly],
#     'FN': [fn_linear, fn_rbf, fn_poly],
#     'Accuracy': [accuracy_linear, accuracy_rbf, accuracy_poly],
#     'Precision': [precision_linear, precision_rbf, precision_poly],
#     'Recall': [recall_linear, recall_rbf, recall_poly],
#     'F1 Score': [f1_score_linear, f1_score_rbf, f1_score_poly],
#     'ConfusionMatrix': [confusion_matrix(actual_labels_linear, predicted_labels_linear),confusion_matrix(actual_labels_rbf, predicted_labels_rbf),confusion_matrix(actual_labels_poly, predicted_labels_poly)]

# }

# df = pd.DataFrame(results)


# df.to_excel('resultados_SVM.xlsx', index=False)

# best_f1_index = df['F1 Score'].idxmax()

# best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']

# plt.figure(figsize=(8, 6))
# sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
#                 xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Best Confusion Matrix')
# plt.show()

# ######################################### -- 10.2 Classification based on Support Vector Machines with PCA -- #####################################################################################
# #print('\n-- Classification based on Support Vector Machines --')

# i3train=np.vstack((trainFeatures_bruno,trainFeatures_dns))
# i3Ctest=np.vstack((testFeatures_bruno,testFeatures_dns))

# #Define a range of components to test
# components_to_test = [5, 10, 15, 20]  
# results=[]
# all_results = []

# for n_components in components_to_test:
#     #Initialize PCA and fit-transform the data
#     pca = PCA(n_components=n_components)
#     i3train_pca = pca.fit_transform(i3train)
#     i3Ctest_pca = pca.transform(i3Ctest)
#     svc = svm.SVC(kernel='linear').fit(i3train, o3trainClass)
#     rbf_svc = svm.SVC(kernel='rbf').fit(i3train, o3trainClass)
#     poly_svc = svm.SVC(kernel='poly',degree=2).fit(i3train, o3trainClass)

#     L1=svc.predict(i3Ctest)
#     L2=rbf_svc.predict(i3Ctest)
#     L3=poly_svc.predict(i3Ctest)

#     tp_linear, fn_linear, tn_linear, fp_linear = 0, 0, 0, 0
#     actual_labels_linear = []
#     predicted_labels_linear = []


#     tp_rbf, fn_rbf, tn_rbf, fp_rbf = 0, 0, 0, 0
#     actual_labels_rbf = []
#     predicted_labels_rbf = []

#     tp_poly, fn_poly, tn_poly, fp_poly = 0, 0, 0, 0
#     actual_labels_poly = []
#     predicted_labels_poly = []

#     nObsTest,nFea=i3Ctest.shape

#     AnomResults={2.0:"Anomaly",0:"OK"} #Bruno é 0 e DNS é 2

#     nObsTest,nFea=i3Ctest.shape

#     for i in range(nObsTest):
#         print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],Classes[L1[i]],Classes[L2[i]],Classes[L3[i]]))
#         actual_labels_linear.append(o3testClass[i][0])
#         actual_labels_rbf.append(o3testClass[i][0])
#         actual_labels_poly.append(o3testClass[i][0])
#         # Linear
#         if AnomResults[L1[i]] == "Anomaly":
#             predicted_labels_linear.append(2.0)  # Predicted as DNS (anomaly)
#             if o3testClass[i][0] == 2:  # DNS class
#                 tp_linear += 1 # True Positive
#             else:  # Marta/Bruno class
#                 fp_linear += 1 # False Positive
#         else: # OK
#             predicted_labels_linear.append(0.0) 
#             if o3testClass[i][0] == 2:
#                 fn_linear += 1  # False Negative
#             else:
#                 tn_linear += 1  # True Negative

#             # RBF
#         if AnomResults[L2[i]] == "Anomaly":
#             predicted_labels_rbf.append(2.0)  # Predicted as DNS (anomaly)
#             if o3testClass[i][0] == 2:  # DNS class
#                 tp_rbf += 1 # True Positive
#             else:  # Marta/Bruno class
#                 fp_rbf += 1 # False Positive
#         else: # OK
#             predicted_labels_rbf.append(0.0) 
#             if o3testClass[i][0] == 2:
#                 fn_rbf += 1  # False Negative
#             else:
#                 tn_rbf += 1  # True Negative
#             # Poly
#         if AnomResults[L3[i]] == "Anomaly":
#             predicted_labels_poly.append(2.0)  # Predicted as DNS (anomaly)
#             if o3testClass[i][0] == 2:  # DNS class
#                 tp_poly += 1 # True Positive
#             else:  # Marta/Bruno class
#                 fp_poly += 1 # False Positive
#         else: # OK
#             predicted_labels_poly.append(0.0) 
#             if o3testClass[i][0] == 2:
#                 fn_poly += 1  # False Negative
#             else:
#                 tn_poly += 1  # True Negative


#     accuracy_linear = ((tp_linear + tn_linear) / nObsTest) * 100
#     precision_linear = (tp_linear / (tp_linear + fp_linear)) * 100 if tp_linear + fp_linear > 0 else 0

#     accuracy_rbf = ((tp_rbf + tn_rbf) / nObsTest) * 100
#     precision_rbf = (tp_rbf / (tp_rbf + fp_rbf)) * 100 if tp_rbf + fp_rbf > 0 else 0

#     accuracy_poly = ((tp_poly + tn_poly) / nObsTest) * 100
#     precision_poly = (tp_poly / (tp_poly + fp_poly)) * 100 if tp_poly + fp_poly > 0 else 0

#     recall_linear = (tp_linear / (tp_linear + fn_linear)) * 100 if tp_linear + fn_linear > 0 else 0
#     recall_rbf = (tp_rbf / (tp_rbf + fn_rbf)) * 100 if tp_rbf + fn_rbf > 0 else 0
#     recall_poly = (tp_poly / (tp_poly + fn_poly)) * 100 if tp_poly + fn_poly > 0 else 0

#     f1_score_linear = (2 * (precision_linear * recall_linear) / (precision_linear + recall_linear)) / 100 if (precision_linear + recall_linear) != 0 else 0
#     f1_score_rbf = (2 * (precision_rbf * recall_rbf) / (precision_rbf + recall_rbf)) / 100 if (precision_rbf + recall_rbf) != 0 else 0
#     f1_score_poly = (2 * (precision_poly * recall_poly) / (precision_poly + recall_poly)) / 100 if (precision_poly + recall_poly) != 0 else 0

#     results = {
#         'Method': ['Linear', 'RBF', 'Poly'],
#         'Number components':n_components,
#         'TP': [tp_linear, tp_rbf, tp_poly],
#         'FP': [fp_linear, fp_rbf, fp_poly],
#         'TN': [tn_linear, tn_rbf, tn_poly],
#         'FN': [fn_linear, fn_rbf, fn_poly],
#         'Accuracy': [accuracy_linear, accuracy_rbf, accuracy_poly],
#         'Precision': [precision_linear, precision_rbf, precision_poly],
#         'Recall': [recall_linear, recall_rbf, recall_poly],
#         'F1 Score': [f1_score_linear, f1_score_rbf, f1_score_poly],
#         'ConfusionMatrix': [confusion_matrix(actual_labels_linear, predicted_labels_linear),confusion_matrix(actual_labels_rbf, predicted_labels_rbf),confusion_matrix(actual_labels_poly, predicted_labels_poly)]

#     }
#     all_results.append(results)


# df = pd.concat([pd.DataFrame(res) for res in all_results], ignore_index=True)

# df.to_excel('resultados_SVM_PCA.xlsx', index=False)

# # Find the index of the row with the best F1 score
# best_f1_index = df['F1 Score'].idxmax()

# best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']


# # Plot the best confusion matrix if it exists
# plt.figure(figsize=(8, 6))
# sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
#                 xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Best Confusion Matrix (Components: {n_components})')
# plt.show()

################################### -- 12 Classification based on Neural Networks without pca -- #########################################################################################################
from sklearn.neural_network import MLPClassifier
# #print('\n-- Classification based on Neural Networks --')
# i3train=np.vstack((trainFeatures_bruno,trainFeatures_dns))
# i3Ctest=np.vstack((testFeatures_bruno,testFeatures_dns))

# scaler = MaxAbsScaler().fit(i3train)
# i3trainN=scaler.transform(i3train)
# i3CtestN=scaler.transform(i3Ctest)


# alpha=1
# max_iter=100000
# clf = MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(20,),max_iter=max_iter)
# clf.fit(i3trainN, o3trainClass)
# LT=clf.predict(i3CtestN)

# #Neural Network
# tp_nn, fn_nn, tn_nn, fp_nn = 0, 0, 0, 0
# acc_nn = []  
# pre_nn = []  
# actual_labels = []
# predicted_labels = []
# results=[]
# nObsTest,nFea=i3CtestN.shape

# for i in range(nObsTest):
#     actual_labels.append(o3testClass[i][0])
#     print(len(actual_labels))
#     #print('Obs: {:2} ({:<8}): Classification->{}'.format(i,Classes[o3testClass[i][0]],Classes[LT[i]]))
#     if LT[i] == o3testClass[i][0]:
#         if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
#             predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
#             tp_nn += 1
#         else:
#             predicted_labels.append(0.0)  # Predicted as Normal
#             fp_nn += 1
#     else:
#         if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
#             predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
#             fn_nn += 1
#         else:
#             predicted_labels.append(0.0)  # Predicted as Normal
#             tn_nn += 1


# accuracy_nn = ((tp_nn + tn_nn) / (tp_nn + tn_nn + fp_nn + fn_nn)) * 100
# precision_nn = (tp_nn / (tp_nn + fp_nn)) * 100 if (tp_nn + fp_nn) != 0 else 0
# recall_nn = (tp_nn / (tp_nn + fn_nn)) * 100 if (tp_nn + fn_nn) != 0 else 0
# f1_score_nn = (2 * (precision_nn * recall_nn)) / (precision_nn + recall_nn) if (precision_nn + recall_nn) != 0 else 0

# results.append({
#     'Accuracy Neural Network': accuracy_nn,
#     'Precision Neural Network': precision_nn,
#     'Recall Neural Network': recall_nn,
#     'F1 Score': f1_score_nn,
#     'TP': tp_nn,
#     'FP': fp_nn,
#     'TN': tn_nn,
#     'FN': fn_nn,
#     'Accuracy': accuracy_nn,
#     'Precision': precision_nn,

# })

# df = pd.DataFrame(results)
# df.to_excel('resultados_redes_neurais.xlsx', index=False)

# confusionMatrix= confusion_matrix(actual_labels, predicted_labels)


# plt.figure(figsize=(8, 6))
# sns.heatmap(confusionMatrix, annot=True, cmap='Blues', fmt='d',
#                 xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Best Confusion Matrix')
# plt.show()


################################### -- 12 Classification based on Neural Networks with pca -- #########################################################################################################

components_to_test = [1,5, 10, 15, 20]

results = []  
i3train=np.vstack((trainFeatures_bruno,trainFeatures_dns))
i3Ctest=np.vstack((testFeatures_bruno,testFeatures_dns))

for n_components in components_to_test:

    pca = PCA(n_components=n_components)
    i3train_pca = pca.fit_transform(i3train)
    i3Ctest_pca = pca.transform(i3Ctest)

    scaler = MaxAbsScaler().fit(i3train_pca)
    i3trainN_pca = scaler.transform(i3train_pca)
    i3CtestN_pca = scaler.transform(i3Ctest_pca)

    alpha = 1
    max_iter = 100000
    clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(20,), max_iter=max_iter)
    clf.fit(i3trainN_pca, o3trainClass)
    LT = clf.predict(i3CtestN_pca)

    tp_nn, fn_nn, tn_nn, fp_nn = 0, 0, 0, 0
    acc_nn = [] 
    pre_nn = []
    actual_labels = []
    predicted_labels = []

    nObsTest, nFea = i3CtestN_pca.shape


    for i in range(nObsTest):
        if LT[i] == o3testClass[i][0]:
            if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
                predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
                tp_nn += 1
            else:
                predicted_labels.append(0.0)  # Predicted as Normal
                fp_nn += 1
        else:
            if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
                predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
                fn_nn += 1
            else:
                predicted_labels.append(0.0)  # Predicted as Normal
                tn_nn += 1



    accuracy_nn = ((tp_nn + tn_nn) / (tp_nn + tn_nn + fp_nn + fn_nn)) * 100
    precision_nn = (tp_nn / (tp_nn + fp_nn)) * 100 if (tp_nn + fp_nn) != 0 else 0
    recall_nn = (tp_nn / (tp_nn + fn_nn)) * 100 if (tp_nn + fn_nn) != 0 else 0
    f1_score_nn = (2 * (precision_nn * recall_nn)) / (precision_nn + recall_nn) if (precision_nn + recall_nn) != 0 else 0

    results.append({
        'Components': n_components,
        'Accuracy Neural Network': accuracy_nn,
        'Precision Neural Network': precision_nn,
        'Recall Neural Network': recall_nn,
        'F1 Score Neural Network': f1_score_nn
    })

df_nn = pd.DataFrame(results)

df_nn.to_excel('resultados_redes_neurais_pca.xlsx', index=False)


# # # Wait for user input before exiting
waitforEnter(fstop=True)