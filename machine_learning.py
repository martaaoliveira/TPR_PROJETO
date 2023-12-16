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
import time
import sys
import warnings
warnings.filterwarnings('ignore')


def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")

## -- 3 -- ##

# features: É a matriz de características (ou atributos) onde cada linha representa uma observação e cada coluna representa um atributo específico.
# oClass: É a matriz de classes correspondentes a cada observação em features.
# f1index e f2index: São os índices dos atributos que você deseja plotar um contra o ou

# f1index representa o índice do atributo que será plotado no eixo x do gráfico.
# f2index representa o índice do atributo que será plotado no eixo y do gráfico.

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


Classes={0:'Marta',1:'Bruno',2:'DNS'}
plt.ion()
nfig=1

## -- 2 -- ##
features_bruno=np.loadtxt("features_bruno.dat")
features_marta=np.loadtxt("features_marta.dat")
features_dns=np.loadtxt("features_dns_tunneling_smart.dat")

#It assigns class labels (0 for Bruno, 1 for Marta, and 2 for dns_tunneling) to the respective datasets
#cada classe vai conter:mean, median and standard deviation  and also the silence periods features(mean median and deviation) and percentis for upload and download 
oClass_bruno=np.ones((len(features_bruno),1))*0
oClass_marta=np.ones((len(features_marta),1))*1
oClass_dns=np.ones((len(features_dns),1))*2


#resulta num conjunto de features que contém todos os dados dessas diferentes fontes combinados verticalmente.
features=np.vstack((features_marta,features_bruno,features_dns))
print("oclass\n")
#um único array oClass que contém todas as classes correspondentes às observações do conjunto de dados combinado features.
oClass=np.vstack((oClass_marta,oClass_bruno,oClass_dns))

scaler = MaxAbsScaler().fit(features)
features_scaled=scaler.transform(features)


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

#UP DOWN
# 0      1        2    3  4  5 |   6   7         8    9  10 11
# MEDIA MEDIANA DESVIO SI SI SI| MEDIA MEDIANA DESVIO SI SI SI

# Upload           Silencio  | Download     Silencio
# m  m  d           m  m  d    m   m  d     m  m d
# 169263 0 488981   62 2 1    4912 0 13927  62 2 1


#print('Train Silence Features Size:',features.shape)


#Para a primeira captura do script dumb de dns tunneling os valores de download sao muito mais elevados do que o comportamento normal -> exfiltração de dados 
#script mais inteligente->mostra que o maximo de download é baixo e os valores estao na mesma faixa 
plt.figure(1)
plt.scatter(features_bruno[:, 0], features_bruno[:, 22], c='blue', label='Bruno')
plt.scatter(features_marta[:, 0], features_marta[:, 22], c='green', label='Marta')
plt.scatter(features_dns[:, 0], features_dns[:, 22], c='red', label='DNS Tunneling')
plt.title('Comparação entre maximo  nPkt vs maximo  nPktDown')
plt.xlabel('maximo  nPkt')  # Rótulo do eixo x
plt.ylabel('maximo  nPktDown')  # Rótulo do eixo y
plt.legend(loc='lower right', title='Classes')


#------------------------------------------------------
#Comparação entre média nPktDown e desvio padrão nPktDown
#quantidade de pacotes de upload varia consideravelmente entre diferentes pontos de medição, o que se reflete no aumento do desvio padrão. Existe uma grande quantidade de 
#downloads anormal relativamente ao bom comportamento
plt.figure(2)
plt.scatter(features_bruno[:, 23], features_bruno[:, 25], c='blue', label='Bruno')
plt.scatter(features_marta[:, 23], features_marta[:, 25], c='green', label='Marta')
plt.scatter(features_dns[:, 23], features_dns[:, 25], c='red', label='DNS Tunneling')
plt.title('Comparação entre média nPktDown e desvio padrão nPktDown')
plt.xlabel('média nPktDown')  # Rótulo do eixo x
plt.ylabel('desvio padrão nPktDown')  # Rótulo do eixo y
plt.legend(loc='lower right', title='Classes')

#------------------------------------------------------

#Comparar media silencio nPktUp com media desvio padrao nPktUp
#Convem ver sempre features relativamente ao download ja que estamos no caso de extração de dados -> mais coerente
#No caso do bom comportamento existe claramente um comportamento humano pois não existe uma linearidade de valores, resultados mais espaçados, maior numero de silencios -> explicar que
#cada um fez um tipo de browing e dessa forma tem um comportamento diferente (eu tinha add block)
#para o caso do dns_tunneling_burro não existe tanta diferença entre a media e o desvio padrao e existe menos silencio -> extração de dados sem limite
plt.figure(3)
# Plotagem dos pontos por classe com cores específicas
plt.scatter(features_bruno[:, 27], features_bruno[:, 28], c='blue', label='Bruno')
plt.scatter(features_marta[:, 27], features_marta[:, 28], c='green', label='Marta')
plt.scatter(features_dns[:, 27], features_dns[:, 28], c='red', label='DNS Tunneling')
plt.title('Comparação entre média de silêncio nPktDown e desvio padrão silêncio nPktDown')
plt.xlabel('média silêncio nPktDown')  # Rótulo do eixo x
plt.ylabel('desvio padrão silêncio nPktDown')  # Rótulo do eixo y
plt.legend(loc='lower right', title='Classes')


#------------------------------------------------------
#Comparação entre desvio BytesDown e percentis 98 BytesDown
# Dispersão dos Dados: Um desvio padrão maior indica maior variabilidade nos dados. 
# Se o desvio padrão for alto e o percentil 98 também for alto, isso sugere que os dados possuem uma ampla dispersão em torno da média e que há uma presença 
# significativa de valores extremos.
# Outliers: Se o percentil 98 for muito maior do que o desvio padrão, isso pode indicar a presença de outliers, ou seja, valores extremamente altos que estão distantes da média. 
# Isso pode ser significativo para entender situações em que ocorrem transferências de grandes volumes de dados em comparação com a maioria dos casos.
# Estabilidade dos Dados: Um desvio padrão pequeno em relação ao percentil 98 sugere uma menor variabilidade nos dados, 
# indicando maior estabilidade nos volumes de dados transferidos na maioria dos casos.
plt.figure(4)
plt.scatter(features_bruno[:, 36], features_bruno[:, 43], c='blue', label='Bruno')
plt.scatter(features_marta[:, 36], features_marta[:, 43], c='green', label='Marta')
plt.scatter(features_dns[:, 36], features_dns[:, 43], c='red', label='DNS Tunneling')
plt.title('Comparação entre desvio BytesDown e percentis 98 BytesDown')
plt.xlabel('desvio BytesDown ')  # Rótulo do eixo x
plt.ylabel('percentis 98 BytesDown')  # Rótulo do eixo y
plt.legend(loc='lower right', title='Classes')


#divisão do conjunto de dados em dados de treino e de teste.

## -- 3 -- ##
#:i
#Define a percentagem dos dados originais que serão usados para TREINO (50% neste caso).
percentage=0.5
#pB, pYT, pM: Calculam o tamanho do conjunto de treino para cada categoria (Browsing, YouTube e Mining) com base na percentagem definida.
pB=int(len(features_bruno)*percentage)
trainFeatures_bruno=features_bruno[:pB,:]
pM=int(len(features_marta)*percentage)
trainFeatures_marta=features_marta[:pM,:]
pD=int(len(features_dns)*percentage)
trainFeatures_dns=features_dns[:pD,:]

#i2train: Build train features of normal behavior
i2train=np.vstack((trainFeatures_bruno,trainFeatures_marta))
o2trainClass=np.vstack((oClass_bruno[:pB],oClass_marta[:pM]))

#:ii
i3Ctrain=np.vstack((trainFeatures_bruno,trainFeatures_marta,trainFeatures_dns))
o3trainClass=np.vstack((oClass_bruno[:pB],oClass_marta[:pM],oClass_dns[:pD]))

#:iii
testFeatures_bruno=features_bruno[pB:,:]
testFeatures_marta=features_marta[pM:,:]
testFeatures_dns=features_dns[pD:,:]
#Fornecer ao modelos os dados de TESTE
i3Atest=np.vstack((testFeatures_bruno,testFeatures_marta,testFeatures_dns))
o3testClass=np.vstack((oClass_bruno[pB:],oClass_marta[pM:],oClass_dns[pD:]))

#                                                                                               ##### -- 7 -- Centroids Distances SEM PCA ######################### ##

# i2train=np.vstack((trainFeatures_bruno,trainFeatures_marta))
# #scaler = MaxAbsScaler().fit(i2train)
# #i2train=scaler.transform(i2train)

# centroids={}
# for c in range(2):  # Only the first two classes (client classes)
#     pClass=(o2trainClass==c).flatten()
#     centroids.update({c:np.mean(i2train[pClass,:],axis=0)})
# #print('All Features Centroids:\n',centroids)

# i3Atest=np.vstack((testFeatures_bruno,testFeatures_marta,testFeatures_dns))
# #i3Atest=scaler.transform(i3Atest)

# print('\n-- Anomaly Detection based on Centroids Distances without PCA --')
# nObsTest,nFea=i3Atest.shape

# # Define a range of threshold values to test
# threshold_values = [0.5, 1.0,1.2, 1.1, 1.5, 2.0,5,10]  # Add more threshold values as needed

# # Initialize lists to store evaluation metrics for each threshold
# threshold_metrics = []

# for AnomalyThreshold in threshold_values:
#     # Initialize variables for TP, FP, TN, FN for the current threshold
#     tp_centroids = 0
#     fp_centroids = 0
#     tn_centroids = 0
#     fn_centroids = 0

#     # Perform anomaly detection based on the current threshold
#     for i in range(nObsTest):
#         x = i3Atest[i]
#         dists = [distance(x, centroids[0]), distance(x, centroids[1])]
#         if min(dists) > AnomalyThreshold:
#             result = "Anomaly"
#             if o3testClass[i][0] == 1:  # Positive class
#                 fn_centroids += 1  # False Negative
#             else:
#                 tn_centroids += 1  # True Negative
#         else:
#             result = "OK"
#             if o3testClass[i][0] == 1:  # Positive class
#                 tp_centroids += 1  # True Positive
#             else:
#                 fp_centroids += 1  # False Positive

#     # Calculate evaluation metrics for the current threshold
#     accuracy_centroids = ((tp_centroids + tn_centroids) / nObsTest) * 100
#     precision_centroids = (tp_centroids / (tp_centroids + fp_centroids)) * 100 if (tp_centroids + fp_centroids) != 0 else 0
#     recall_centroids = (tp_centroids / (tp_centroids + fn_centroids)) * 100 if (tp_centroids + fn_centroids) != 0 else 0
#     f1_score_centroids = (2 * (precision_centroids * recall_centroids) / (precision_centroids + recall_centroids))/100 if (precision_centroids + recall_centroids) != 0 else 0

#     # Store metrics for the current threshold in the list
#     threshold_metrics.append({
#         'AnomalyThreshold': AnomalyThreshold,
#         'TP': tp_centroids,
#         'FP': fp_centroids,
#         'TN': tn_centroids,
#         'FN': fn_centroids,
#         'Accuracy': accuracy_centroids,
#         'Precision': precision_centroids,
#         'Recall': recall_centroids,
#         'F1 Score': f1_score_centroids
#     })

# # Print metrics for each threshold
# for metric in threshold_metrics:
#     print(f"Anomaly Threshold: {metric['AnomalyThreshold']}")
#     print(f"True Positives (TP): {metric['TP']}")
#     print(f"False Negatives (FN): {metric['FN']}")
#     print(f"False Positives (FP): {metric['FP']}")
#     print(f"True Negatives (TN): {metric['TN']}")
#     print(f"Accuracy: {metric['Accuracy']}")
#     print(f"Precision: {metric['Precision']}")
#     print(f"Recall: {metric['Recall']}")
#     print(f"F1 Score: {metric['F1 Score']}")
#     print("-------------------")


# #                                                                                    ######## -- 7.2 -- Centroids Distances Com PCA ######################### ##
# # Define a range of components to test
# components_to_test = [10, 15, 20]  # Adjust as needed
# print('\n-- Anomaly Detection based on Centroids Distances with PCA --')

# for n_components in components_to_test:
#     i2train = np.vstack((trainFeatures_bruno, trainFeatures_marta))
#     #scaler = MaxAbsScaler().fit(i2train)
#     #i2train = scaler.transform(i2train)

#     # Initialize PCA and fit-transform the data
#     pca = PCA(n_components=n_components)
#     i2train_pca = pca.fit_transform(i2train)

#     centroids = {}
#     for c in range(2):  # Only the first two classes (client classes)
#         pClass = (o2trainClass == c).flatten()
#         centroids.update({c: np.mean(i2train_pca[pClass, :], axis=0)})

#     i3Atest = np.vstack((testFeatures_bruno, testFeatures_marta, testFeatures_dns))
#     #i3Atest = scaler.transform(i3Atest)
#     i3Atest_pca = pca.transform(i3Atest)

#     print(f'\n-- Anomaly Detection based on Centroids Distances (Components: {n_components}) --')
#     nObsTest, nFea = i3Atest_pca.shape

#     # Define a range of threshold values to test
#     threshold_values = [0.5,1,2]  # Add more threshold values as needed

#     # Initialize lists to store evaluation metrics for each threshold
#     threshold_metrics = []

#     for AnomalyThreshold in threshold_values:
#         # Initialize variables for TP, FP, TN, FN for the current threshold
#         tp_centroids = 0
#         fp_centroids = 0
#         tn_centroids = 0
#         fn_centroids = 0

#         # Perform anomaly detection based on the current threshold
#         for i in range(nObsTest):
#             x = i3Atest_pca[i]
#             dists = [distance(x, centroids[0]), distance(x, centroids[1])]
#             if min(dists) > AnomalyThreshold:
#                 result = "Anomaly"
#                 if o3testClass[i][0] == 1:  # Positive class
#                     fn_centroids += 1  # False Negative
#                 else:
#                     tn_centroids += 1  # True Negative
#             else:
#                 result = "OK"
#                 if o3testClass[i][0] == 1:  # Positive class
#                     tp_centroids += 1  # True Positive
#                 else:
#                     fp_centroids += 1  # False Positive

#         # Calculate evaluation metrics for the current threshold
#         accuracy_centroids = ((tp_centroids + tn_centroids) / nObsTest) * 100
#         precision_centroids = (tp_centroids / (tp_centroids + fp_centroids)) * 100 if (
#                     tp_centroids + fp_centroids) != 0 else 0
#         recall_centroids = (tp_centroids / (tp_centroids + fn_centroids)) * 100 if (
#                     tp_centroids + fn_centroids) != 0 else 0
#         f1_score_centroids = (2 * (precision_centroids * recall_centroids) / (
#                     precision_centroids + recall_centroids)) / 100 if (
#                                         precision_centroids + recall_centroids) != 0 else 0

#         # Store metrics for the current threshold in the list
#         threshold_metrics.append({
#             'AnomalyThreshold': AnomalyThreshold,
#             'TP': tp_centroids,
#             'FP': fp_centroids,
#             'TN': tn_centroids,
#             'FN': fn_centroids,
#             'Accuracy': accuracy_centroids,
#             'Precision': precision_centroids,
#             'Recall': recall_centroids,
#             'F1 Score': f1_score_centroids
#         })

#     # Print metrics for each threshold
#     for metric in threshold_metrics:
#         print(f"Anomaly Threshold: {metric['AnomalyThreshold']}")
#         print(f"True Positives (TP): {metric['TP']}")
#         print(f"False Negatives (FN): {metric['FN']}")
#         print(f"False Positives (FP): {metric['FP']}")
#         print(f"True Negatives (TN): {metric['TN']}")
#         print(f"Accuracy: {metric['Accuracy']}")
#         print(f"Precision: {metric['Precision']}")
#         print(f"Recall: {metric['Recall']}")
#         print(f"F1 Score: {metric['F1 Score']}")
#         print("-------------------")


#                                                                           # -- 8 -- Anomaly Detection based on One Class Support Vector Machines WITHOUT PCA ###############################
# print('\n-- Anomaly Detection based on One Class Support Vector Machines--')
# i2train=np.vstack((trainFeatures_marta,trainFeatures_bruno))
# i3Atest=np.vstack((testFeatures_bruno,testFeatures_marta,testFeatures_dns))

# nu=0.5
# ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear',nu=nu).fit(i2train)
# rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf',nu=nu).fit(i2train)
# poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',nu=nu,degree=2).fit(i2train)

# L1=ocsvm.predict(i3Atest)
# L2=rbf_ocsvm.predict(i3Atest)
# L3=poly_ocsvm.predict(i3Atest)

# AnomResults={-1:"Anomaly",1:"OK"}

# # 
# tp_linear, fn_linear, tn_linear, fp_linear = 0, 0, 0, 0
# tp_rbf, fn_rbf, tn_rbf, fp_rbf = 0, 0, 0, 0
# tp_poly, fn_poly, tn_poly, fp_poly = 0, 0, 0, 0


# nObsTest,nFea=i3Atest.shape
# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    
#     # Linear
#     if AnomResults[L1[i]] == "Anomaly":
#         if o3testClass[i][0] == 1:  # Positive class
#             tp_linear += 1
#         else:  # Negative class
#             fp_linear += 1
#     else:
#         if o3testClass[i][0] == 1:
#             fn_linear += 1
#         else:
#             tn_linear += 1

#     # RBF
#     if AnomResults[L2[i]] == "Anomaly":
#         if o3testClass[i][0] == 1:  # Positive class
#             tp_rbf += 1
#         else:  # Negative class
#             fp_rbf += 1
#     else:
#         if o3testClass[i][0] == 1:
#             fn_rbf += 1
#         else:
#             tn_rbf += 1

#     # Poly
#     if AnomResults[L3[i]] == "Anomaly":
#         if o3testClass[i][0] == 1:  # Positive class
#             tp_poly += 1
#         else:  # Negative class
#             fp_poly += 1
#     else:
#         if o3testClass[i][0] == 1:
#             fn_poly += 1
#         else:
#             tn_poly += 1


# total_samples = nObsTest  # O número total de amostras

# accuracy_linear = ((tp_linear + tn_linear) / total_samples) * 100
# precision_linear = (tp_linear / (tp_linear + fp_linear)) * 100 if tp_linear + fp_linear > 0 else 0

# accuracy_rbf = ((tp_rbf + tn_rbf) / total_samples) * 100
# precision_rbf = (tp_rbf / (tp_rbf + fp_rbf)) * 100 if tp_rbf + fp_rbf > 0 else 0

# accuracy_poly = ((tp_poly + tn_poly) / total_samples) * 100
# precision_poly = (tp_poly / (tp_poly + fp_poly)) * 100 if tp_poly + fp_poly > 0 else 0


# print("Metrics for Linear Kernel:")
# print(f"True Positives (TP): {tp_linear}")
# print(f"False Positives (FP): {fp_linear}")
# print(f"True Negatives (TN): {tn_linear}")
# print(f"False Negatives (FN): {fn_linear}\n")
# print(f"Accuracy: {accuracy_linear}")
# print(f"Precision: {precision_linear}\n")

# print("Metrics for RBF Kernel:")
# print(f"True Positives (TP): {tp_rbf}")
# print(f"False Positives (FP): {fp_rbf}")
# print(f"True Negatives (TN): {tn_rbf}")
# print(f"False Negatives (FN): {fn_rbf}\n")
# print(f"Accuracy: {accuracy_rbf}")
# print(f"Precision: {precision_rbf}\n")


# print("Metrics for Poly Kernel:")
# print(f"True Positives (TP): {tp_poly}")
# print(f"False Positives (FP): {fp_poly}")
# print(f"True Negatives (TN): {tn_poly}")
# print(f"False Negatives (FN): {fn_poly}\n")
# print(f"Accuracy: {accuracy_poly}")
# print(f"Precision: {precision_poly}\n")




#                                                                       # -- 8.2 -- Anomaly Detection based on One Class Support Vector Machines with pca###############################
# pca = PCA(n_components=10)  # Número de componentes principais
# i2train_pca = pca.fit_transform(i2train)
# i3Atest_pca = pca.transform(i3Atest)

# n_components_list = [1,5,10,15,16,17,18,19,20,21]
# tp, fn, tn, fp = 0, 0, 0, 0

# # Loop pelos diferentes números de componentes PCA
# for n_components in n_components_list:
#     # Reduzindo a dimensionalidade com PCA
#     pca = PCA(n_components=n_components)
#     i2train_pca = pca.fit_transform(i2train)
#     i3Atest_pca = pca.transform(i3Atest)
    
#     # Treinando o modelo OneClassSVM
#     nu = 0.5
#     ocsvm = OneClassSVM(gamma='scale', kernel='linear', nu=nu).fit(i2train_pca)
#     L = ocsvm.predict(i3Atest_pca)
    
#     # Calculando as métricas
#     tp, fn, tn, fp = 0, 0, 0, 0
#     for i in range(len(L)):
#         if L[i] == 1 and o3testClass[i][0] == 1:
#             tp += 1
#     # Condição para falso positivo (FP)
#         elif L[i] == 1 and o3testClass[i][0] != 1:
#             fp += 1
#     # Condição para verdadeiro negativo (TN)
#         elif L[i] != 1 and o3testClass[i][0] != 1:
#             tn += 1
#     # Condição para falso negativo (FN)
#         elif L[i] != 1 and o3testClass[i][0] == 1:
#             fn += 1
        
#     accuracy = ((tp + tn) / len(L)) * 100
#     precision = (tp / (tp + fp)) * 100 if tp + fp > 0 else 0
    
#     # Imprimindo os resultados para cada número de componentes PCA
#     print(f"Metrics for {n_components} PCA Components:")
#     print(f"True Positives (TP): {tp}")
#     print(f"False Positives (FP): {fp}")
#     print(f"True Negatives (TN): {tn}")
#     print(f"False Negatives (FN): {fn}\n")
#     print(f"Accuracy: {accuracy}")
#     print(f"Precision: {precision}\n")



#                                                                           ###### -- 10 Classification based on Support Vector Machines without PCA -- #####################################################################################
# print('\n-- Classification based on Support Vector Machines --')

# i3train=np.vstack((trainFeatures_bruno,trainFeatures_marta,trainFeatures_dns))
# i3Ctest=np.vstack((testFeatures_bruno,testFeatures_marta,testFeatures_dns))


# svc = svm.SVC(kernel='linear').fit(i3train, o3trainClass)
# rbf_svc = svm.SVC(kernel='rbf').fit(i3train, o3trainClass)
# poly_svc = svm.SVC(kernel='poly',degree=2).fit(i3train, o3trainClass)

# L1=svc.predict(i3Ctest)
# L2=rbf_svc.predict(i3Ctest)
# L3=poly_svc.predict(i3Ctest)
# print('\n')

# # Linear SVM
# tp_svm_linear, fn_svm_linear, tn_svm_linear, fp_svm_linear = 0, 0, 0, 0
# # RBF SVM
# tp_svm_rbf, fn_svm_rbf, tn_svm_rbf, fp_svm_rbf = 0, 0, 0, 0
# # Poly SVM
# tp_svm_poly, fn_svm_poly, tn_svm_poly, fp_svm_poly = 0, 0, 0, 0

# nObsTest,nFea=i3Ctest.shape

# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],Classes[L1[i]],Classes[L2[i]],Classes[L3[i]]))

#     # Linear SVM

#     if Classes[L1[i]] == Classes[o3testClass[i][0]]:
#         if Classes[L1[i]] == 'DNS':  # Positive class
#             tp_svm_linear += 1
#         else:  # Negative class
#             tn_svm_linear += 1
#     else:
#         if Classes[L1[i]] == 'DNS':
#             fn_svm_linear += 1
#         else:
#             fp_svm_linear += 1

#     # RBF SVM (Repetir a lógica similar para o kernel RBF)
#     if Classes[L2[i]] == Classes[o3testClass[i][0]]:
#         if Classes[L2[i]] == 'DNS':  # Positive class
#             tp_svm_rbf += 1
#         else:  # Negative class
#             tn_svm_rbf += 1
#     else:
#         if Classes[L2[i]] == 'DNS':
#             fn_svm_rbf += 1
#         else:
#             fp_svm_rbf += 1

#     # Poly SVM (Repetir a lógica similar para o kernel Poly)
#     if Classes[L3[i]] == Classes[o3testClass[i][0]]:
#         if Classes[L3[i]] == 'DNS':  # Positive class
#             tp_svm_poly += 1
#         else:  # Negative class
#             tn_svm_poly += 1
#     else:
#         if Classes[L3[i]] == 'DNS':
#             fn_svm_poly += 1
#         else:
#             fp_svm_poly += 1


# accuracy_svm_linear = ((tp_svm_linear + tn_svm_linear) / (tp_svm_linear + tn_svm_linear + fp_svm_linear + fn_svm_linear))*100
# precision_svm_linear = (tp_svm_linear / (tp_svm_linear + fp_svm_linear))*100

# accuracy_svm_rbf = ((tp_svm_rbf + tn_svm_rbf) / (tp_svm_rbf + tn_svm_rbf + fp_svm_rbf + fn_svm_rbf))*100
# precision_svm_rbf = (tp_svm_rbf / (tp_svm_rbf + fp_svm_rbf))*100

# accuracy_svm_poly = ((tp_svm_poly + tn_svm_poly) / (tp_svm_poly + tn_svm_poly + fp_svm_poly + fn_svm_poly))*100
# precision_svm_poly = (tp_svm_poly / (tp_svm_poly + fp_svm_poly))*100




# # Imprimir as métricas para cada modelo SVM
# print("Metrics for Linear SVM:")
# print(f"True Positives (TP): {tp_svm_linear}")
# print(f"False Positives (FP): {fp_svm_linear}")
# print(f"True Negatives (TN): {tn_svm_linear}")
# print(f"False Negatives (FN): {fn_svm_linear}\n")
# print(f"Accuracy: {accuracy_svm_linear}")
# print(f"Precision: {precision_svm_linear}\n")


# print("Metrics for RBF SVM:")
# print(f"True Positives (TP): {tp_svm_rbf}")
# print(f"False Positives (FP): {fp_svm_rbf}")
# print(f"True Negatives (TN): {tn_svm_rbf}")
# print(f"False Negatives (FN): {fn_svm_rbf}\n")
# print(f"Accuracy: {accuracy_svm_rbf}")
# print(f"Precision: {precision_svm_rbf}\n")



# print("Metrics for Poly SVM:")
# print(f"True Positives (TP): {tp_svm_poly}")
# print(f"False Positives (FP): {fp_svm_poly}")
# print(f"True Negatives (TN): {tn_svm_poly}")
# print(f"False Negatives (FN): {fn_svm_poly}\n")
# print(f"Accuracy: {accuracy_svm_poly}")
# print(f"Precision: {precision_svm_poly}\n")





#                                                                               ###### -- 10.2 Classification based on Support Vector Machines with PCA -- #####################################################################################
# print('\n-- Classification based on Support Vector Machines --')

# i3train=np.vstack((trainFeatures_bruno,trainFeatures_marta,trainFeatures_dns))
# i3Ctest=np.vstack((testFeatures_bruno,testFeatures_marta,testFeatures_dns))

# # Define a range of components to test
# components_to_test = [5, 10, 15, 20]  # Adjust as needed

# for n_components in components_to_test:
#     # Initialize PCA and fit-transform the data
#     pca = PCA(n_components=n_components)
#     i3train_pca = pca.fit_transform(i3train)
#     i3Ctest_pca = pca.transform(i3Ctest)
#     svc = svm.SVC(kernel='linear').fit(i3train, o3trainClass)
#     rbf_svc = svm.SVC(kernel='rbf').fit(i3train, o3trainClass)
#     poly_svc = svm.SVC(kernel='poly',degree=2).fit(i3train, o3trainClass)

#     L1=svc.predict(i3Ctest)
#     L2=rbf_svc.predict(i3Ctest)
#     L3=poly_svc.predict(i3Ctest)
#     print('\n')

#     # Linear SVM
#     tp_svm_linear, fn_svm_linear, tn_svm_linear, fp_svm_linear = 0, 0, 0, 0
#     # RBF SVM
#     tp_svm_rbf, fn_svm_rbf, tn_svm_rbf, fp_svm_rbf = 0, 0, 0, 0
#     # Poly SVM
#     tp_svm_poly, fn_svm_poly, tn_svm_poly, fp_svm_poly = 0, 0, 0, 0

#     nObsTest,nFea=i3Ctest.shape

#     for i in range(nObsTest):
#         #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],Classes[L1[i]],Classes[L2[i]],Classes[L3[i]]))
#         # Linear SVM
#         if Classes[L1[i]] == Classes[o3testClass[i][0]]:
#             if Classes[L1[i]] == 'DNS':  # Positive class
#                 tp_svm_linear += 1
#             else:  # Negative class
#                 tn_svm_linear += 1
#         else:
#             if Classes[L1[i]] == 'DNS':
#                 fn_svm_linear += 1
#             else:
#                 fp_svm_linear += 1

#         # RBF SVM (Repetir a lógica similar para o kernel RBF)
#         if Classes[L2[i]] == Classes[o3testClass[i][0]]:
#             if Classes[L2[i]] == 'DNS':  # Positive class
#                 tp_svm_rbf += 1
#             else:  # Negative class
#                 tn_svm_rbf += 1
#         else:
#             if Classes[L2[i]] == 'DNS':
#                 fn_svm_rbf += 1
#             else:
#                 fp_svm_rbf += 1

#         # Poly SVM (Repetir a lógica similar para o kernel Poly)
#         if Classes[L3[i]] == Classes[o3testClass[i][0]]:
#             if Classes[L3[i]] == 'DNS':  # Positive class
#                 tp_svm_poly += 1
#             else:  # Negative class
#                 tn_svm_poly += 1
#         else:
#             if Classes[L3[i]] == 'DNS':
#                 fn_svm_poly += 1
#             else:
#                 fp_svm_poly += 1


#      # Calculate Accuracy for Linear SVM
#     accuracy_svm_linear = ((tp_svm_linear + tn_svm_linear) / (tp_svm_linear + tn_svm_linear + fp_svm_linear + fn_svm_linear)) * 100

#     # Calculate Accuracy for RBF SVM
#     accuracy_svm_rbf = ((tp_svm_rbf + tn_svm_rbf) / (tp_svm_rbf + tn_svm_rbf + fp_svm_rbf + fn_svm_rbf)) * 100

#     # Calculate Accuracy for Poly SVM
#     accuracy_svm_poly = ((tp_svm_poly + tn_svm_poly) / (tp_svm_poly + tn_svm_poly + fp_svm_poly + fn_svm_poly)) * 100

#     precision_svm_linear = (tp_svm_linear / (tp_svm_linear + fp_svm_linear)) if (tp_svm_linear + fp_svm_linear) != 0 else 0
#     recall_svm_linear = (tp_svm_linear / (tp_svm_linear + fn_svm_linear)) if (tp_svm_linear + fn_svm_linear) != 0 else 0
#     f1_score_svm_linear = 2 * (precision_svm_linear * recall_svm_linear) / (precision_svm_linear + recall_svm_linear) if (precision_svm_linear + recall_svm_linear) != 0 else 0

#     # Calculate F1 score for RBF SVM
#     precision_svm_rbf = (tp_svm_rbf / (tp_svm_rbf + fp_svm_rbf)) if (tp_svm_rbf + fp_svm_rbf) != 0 else 0
#     recall_svm_rbf = (tp_svm_rbf / (tp_svm_rbf + fn_svm_rbf)) if (tp_svm_rbf + fn_svm_rbf) != 0 else 0
#     f1_score_svm_rbf = 2 * (precision_svm_rbf * recall_svm_rbf) / (precision_svm_rbf + recall_svm_rbf) if (precision_svm_rbf + recall_svm_rbf) != 0 else 0

#     # Calculate F1 score for Poly SVM
#     precision_svm_poly = (tp_svm_poly / (tp_svm_poly + fp_svm_poly)) if (tp_svm_poly + fp_svm_poly) != 0 else 0
#     recall_svm_poly = (tp_svm_poly / (tp_svm_poly + fn_svm_poly)) if (tp_svm_poly + fn_svm_poly) != 0 else 0
#     f1_score_svm_poly = 2 * (precision_svm_poly * recall_svm_poly) / (precision_svm_poly + recall_svm_poly) if (precision_svm_poly + recall_svm_poly) != 0 else 0

#     # Print your metrics including F1 score for each SVM
#     print(f"Metrics for Linear SVM with {n_components} components:")
#     print(f"True Positives (TP): {tp_svm_linear}")
#     print(f"False Positives (FP): {fp_svm_linear}")
#     print(f"True Negatives (TN): {tn_svm_linear}")
#     print(f"False Negatives (FN): {fn_svm_linear}")
#     print(f"Accuracy: {accuracy_svm_linear}")
#     print(f"Precision: {precision_svm_linear}")
#     print(f"Recall: {recall_svm_linear}")
#     print(f"F1 Score: {f1_score_svm_linear}\n")

#     print(f"Metrics for RBF SVM with {n_components} components:")
#     print(f"True Positives (TP): {tp_svm_rbf}")
#     print(f"False Positives (FP): {fp_svm_rbf}")
#     print(f"True Negatives (TN): {tn_svm_rbf}")
#     print(f"False Negatives (FN): {fn_svm_rbf}")
#     print(f"Accuracy: {accuracy_svm_rbf}")
#     print(f"Precision: {precision_svm_rbf}")
#     print(f"Recall: {recall_svm_rbf}")
#     print(f"F1 Score: {f1_score_svm_rbf}\n")

#     print(f"Metrics for Poly SVM with {n_components} components:")
#     print(f"True Positives (TP): {tp_svm_poly}")
#     print(f"False Positives (FP): {fp_svm_poly}")
#     print(f"True Negatives (TN): {tn_svm_poly}")
#     print(f"False Negatives (FN): {fn_svm_poly}")
#     print(f"Accuracy: {accuracy_svm_poly}")
#     print(f"Precision: {precision_svm_poly}")
#     print(f"Recall: {recall_svm_poly}")
#     print(f"F1 Score: {f1_score_svm_poly}\n")




#                                                                       ######## -- 12 Classification based on Neural Networks without pca -- #########################################################################################################
# from sklearn.neural_network import MLPClassifier
# print('\n-- Classification based on Neural Networks --')
# i3train=np.vstack((trainFeatures_bruno,trainFeatures_marta,trainFeatures_dns))
# i3Ctest=np.vstack((testFeatures_bruno,testFeatures_marta,testFeatures_dns))

# scaler = MaxAbsScaler().fit(i3train)
# i3trainN=scaler.transform(i3train)
# i3CtestN=scaler.transform(i3Ctest)


# alpha=1
# max_iter=100000
# clf = MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(20,),max_iter=max_iter)
# clf.fit(i3trainN, o3trainClass)
# LT=clf.predict(i3CtestN)

# # Neural Network
# tp_nn, fn_nn, tn_nn, fp_nn = 0, 0, 0, 0
# acc_nn = []  # Lista para armazenar a acurácia
# pre_nn = []  # Lista para armazenar a precisão

# nObsTest,nFea=i3CtestN.shape
# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Classification->{}'.format(i,Classes[o3testClass[i][0]],Classes[LT[i]]))

#     if LT[i] == o3testClass[i][0]:
#         if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
#             tp_nn += 1
#         else:
#             tn_nn += 1
#     else:
#         if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
#             fp_nn += 1
#         else:
#             fn_nn += 1


# # Calcular a acurácia
# accuracy_nn = ((tp_nn + tn_nn) / (tp_nn + tn_nn + fp_nn + fn_nn))*100

# # Calcular a precisão
# precision_nn = (tp_nn / (tp_nn + fp_nn))*100

# # Adicionar os valores de acurácia e precisão às listas correspondentes
# acc_nn.append(accuracy_nn)
# pre_nn.append(precision_nn)



# # Imprimir os resultados
# print("Metrics for Neural Network:")
# print(f"True Positives (TP): {tp_nn}")
# print(f"False Positives (FP): {fp_nn}")
# print(f"True Negatives (TN): {tn_nn}")
# print(f"False Negatives (FN): {fn_nn}")
# print(f"Accuracy: {accuracy_nn:.2f}")
# print(f"Precision: {precision_nn:.2f}")

# # Wait for user input before exiting
waitforEnter(fstop=True)