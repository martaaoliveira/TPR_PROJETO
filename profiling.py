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
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import seaborn as sns
import time
import sys
import warnings
import pandas as pd
import os

warnings.filterwarnings('ignore')

def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")

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


######################################### -- 8 -- Anomaly Detection based on One Class Support Vector Machines WITHOUT PCA ###############################
def one_class_svm(trainFeatures, testFeatures_normal, testFeatures_dns, o3testClass,name_excel):
    tp_linear, fn_linear, tn_linear, fp_linear = 0, 0, 0, 0
    tp_rbf, fn_rbf, tn_rbf, fp_rbf = 0, 0, 0, 0
    tp_poly, fn_poly, tn_poly, fp_poly = 0, 0, 0, 0
    
    i2train = np.vstack(trainFeatures)
    i3Atest = np.vstack((testFeatures_normal, testFeatures_dns))
    
    results = []
    AnomResults = {-1: "Anomaly", 1: "OK"}
    nObsTest, nFea = i3Atest.shape

    nu = 0.5
    ocsvm = svm.OneClassSVM(gamma='scale', kernel='linear', nu=nu).fit(i2train)
    rbf_ocsvm = svm.OneClassSVM(gamma='scale', kernel='rbf', nu=nu).fit(i2train)
    poly_ocsvm = svm.OneClassSVM(gamma='scale', kernel='poly', nu=nu, degree=2).fit(i2train)

    L1 = ocsvm.predict(i3Atest)
    L2 = rbf_ocsvm.predict(i3Atest)
    L3 = poly_ocsvm.predict(i3Atest)

    actual_labels_linear = []
    predicted_labels_linear = []

    actual_labels_rbf = []
    predicted_labels_rbf = []

    actual_labels_poly = []
    predicted_labels_poly = []

    for i in range(nObsTest):
        # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
        actual_labels_linear.append(o3testClass[i][0])
        actual_labels_rbf.append(o3testClass[i][0])
        actual_labels_poly.append(o3testClass[i][0])

        # Linear
        if AnomResults[L1[i]] == "Anomaly":
            predicted_labels_linear.append(2.0)
            if o3testClass[i][0] == 2:
                tp_linear += 1
            else:
                fp_linear += 1
        else:
            predicted_labels_linear.append(0.0)
            if o3testClass[i][0] == 2:
                fn_linear += 1
            else:
                tn_linear += 1

        # RBF
        if AnomResults[L2[i]] == "Anomaly":
            predicted_labels_rbf.append(2.0)
            if o3testClass[i][0] == 2:
                tp_rbf += 1
            else:
                fp_rbf += 1
        else:
            predicted_labels_rbf.append(0.0)
            if o3testClass[i][0] == 2:
                fn_rbf += 1
            else:
                tn_rbf += 1


        # Poly
        if AnomResults[L3[i]] == "Anomaly":
            predicted_labels_poly.append(2.0)
            if o3testClass[i][0] == 2:
                tp_poly += 1
            else:
                fp_poly += 1
        else:
            predicted_labels_poly.append(0.0)
            if o3testClass[i][0] == 2:
                fn_poly += 1
            else:
                tn_poly += 1

    accuracy_linear = ((tp_linear + tn_linear) / nObsTest) * 100
    precision_linear = (tp_linear / (tp_linear + fp_linear)) * 100 if tp_linear + fp_linear > 0 else 0
    recall_linear = (tp_linear / (tp_linear + fn_linear)) * 100 if tp_linear + fn_linear > 0 else 0
    f1_score_linear = (2 * (precision_linear * recall_linear) / (precision_linear + recall_linear))  if (precision_linear + recall_linear) != 0 else 0

    accuracy_rbf = ((tp_rbf + tn_rbf) / nObsTest) * 100
    precision_rbf = (tp_rbf / (tp_rbf + fp_rbf)) * 100 if tp_rbf + fp_rbf > 0 else 0
    recall_rbf = (tp_rbf / (tp_rbf + fn_rbf)) * 100 if tp_rbf + fn_rbf > 0 else 0
    f1_score_rbf = (2 * (precision_rbf * recall_rbf) / (precision_rbf + recall_rbf))  if (precision_rbf + recall_rbf) != 0 else 0

    accuracy_poly = ((tp_poly + tn_poly) / nObsTest) * 100
    precision_poly = (tp_poly / (tp_poly + fp_poly)) * 100 if tp_poly + fp_poly > 0 else 0
    recall_poly = (tp_poly / (tp_poly + fn_poly)) * 100 if tp_poly + fn_poly > 0 else 0
    f1_score_poly = (2 * (precision_poly * recall_poly) / (precision_poly + recall_poly))  if (precision_poly + recall_poly) != 0 else 0

    results = {
        'Method': ['Linear', 'RBF', 'Poly'],
        'TP': [tp_linear, tp_rbf, tp_poly],
        'FP': [fp_linear, fp_rbf, fp_poly],
        'TN': [tn_linear, tn_rbf, tn_poly],
        'FN': [fn_linear, fn_rbf, fn_poly],
        'Accuracy': [accuracy_linear, accuracy_rbf, accuracy_poly],
        'Precision': [precision_linear, precision_rbf, precision_poly],
        'Recall': [recall_linear, recall_rbf, recall_poly],
        'F1 Score': [f1_score_linear, f1_score_rbf, f1_score_poly],
        'Labels': [predicted_labels_linear, predicted_labels_rbf, predicted_labels_poly],
        'ConfusionMatrix': [
            confusion_matrix(actual_labels_linear, predicted_labels_linear),
            confusion_matrix(actual_labels_rbf, predicted_labels_rbf),
            confusion_matrix(actual_labels_poly, predicted_labels_poly)
        ]
    }

    # Create a DataFrame from the results list
    df = pd.DataFrame(results)

    # Save the DataFrame to an Excel file
    # df.to_excel(name_excel+'resultados_OneClassSVM.xlsx', index=False)
    df.to_excel(os.path.join('resultados_script_dumb', f'{name_excel}_resultados_OneClassSVM.xlsx'), index=False)

    # Find the index of the row with the best F1 score
    best_f1_index = df['F1 Score'].idxmax()
    best_f1_value = df['F1 Score'].max()

    best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']
    best_kernel = df.loc[best_f1_index,'Method']
    best_labels = df.loc[best_f1_index,'Labels']

    # Plot the best confusion matrix if it exists
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['Normal', 'DNS TUNNEL'], yticklabels=['Normal', 'DNS TUNNEL'])
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.title(f'Best Confusion Matrix One Class SVM \n Best Kernel: {best_kernel}')
    plt.show()

    print("F1 Score OSVM: ", best_f1_value)

    return best_labels



##################################################################################### -- 8.2 -- Anomaly Detection based on One Class Support Vector Machines with pca###############################
def one_class_svm_with_pca(trainFeatures, testFeatures_normal, testFeatures_dns, o3testClass,name_excel):
    n_components_list = [1, 5, 10, 15]

    results = []
    all_results = []

    for n_components in n_components_list:
        pca = PCA(n_components=n_components)
        i2train_pca = pca.fit_transform(trainFeatures)
        i3Atest_pca = pca.transform(np.vstack((testFeatures_normal, testFeatures_dns)))

        nu = 0.5
        ocsvm = svm.OneClassSVM(gamma='scale', kernel='linear', nu=nu).fit(i2train_pca)
        rbf_ocsvm = svm.OneClassSVM(gamma='scale', kernel='rbf', nu=nu).fit(i2train_pca)
        poly_ocsvm = svm.OneClassSVM(gamma='scale', kernel='poly', nu=nu, degree=2).fit(i2train_pca)

        L1 = ocsvm.predict(i3Atest_pca)
        L2 = rbf_ocsvm.predict(i3Atest_pca)
        L3 = poly_ocsvm.predict(i3Atest_pca)

        tp_linear, fn_linear, tn_linear, fp_linear = 0, 0, 0, 0
        tp_rbf, fn_rbf, tn_rbf, fp_rbf = 0, 0, 0, 0
        tp_poly, fn_poly, tn_poly, fp_poly = 0, 0, 0, 0

        AnomResults = {-1: "Anomaly", 1: "OK"}
        actual_labels_linear = []
        predicted_labels_linear = []

        actual_labels_rbf = []
        predicted_labels_rbf = []

        actual_labels_poly = []
        predicted_labels_poly = []

        nObsTest, nFea = i3Atest_pca.shape
        for i in range(nObsTest):
            actual_labels_linear.append(o3testClass[i][0])
            actual_labels_rbf.append(o3testClass[i][0])
            actual_labels_poly.append(o3testClass[i][0])

            # Linear
            if AnomResults[L1[i]] == "Anomaly":
                predicted_labels_linear.append(2.0)
                if o3testClass[i][0] == 2:
                    tp_linear += 1
                else:
                    fp_linear += 1
            else:
                predicted_labels_linear.append(0.0)
                if o3testClass[i][0] == 2:
                    fn_linear += 1
                else:
                    tn_linear += 1

            # RBF
            if AnomResults[L2[i]] == "Anomaly":
                predicted_labels_rbf.append(2.0)
                if o3testClass[i][0] == 2:
                    tp_rbf += 1
                else:
                    fp_rbf += 1
            else:
                predicted_labels_rbf.append(0.0)
                if o3testClass[i][0] == 2:
                    fn_rbf += 1
                else:
                    tn_rbf += 1


            # Poly
            if AnomResults[L3[i]] == "Anomaly":
                predicted_labels_poly.append(2.0)
                if o3testClass[i][0] == 2:
                    tp_poly += 1
                else:
                    fp_poly += 1
            else:
                predicted_labels_poly.append(0.0)
                if o3testClass[i][0] == 2:
                    fn_poly += 1
                else:
                    tn_poly += 1

        accuracy_linear = ((tp_linear + tn_linear) / nObsTest) * 100
        precision_linear = (tp_linear / (tp_linear + fp_linear)) * 100 if tp_linear + fp_linear > 0 else 0

        accuracy_rbf = ((tp_rbf + tn_rbf) / nObsTest) * 100
        precision_rbf = (tp_rbf / (tp_rbf + fp_rbf)) * 100 if tp_rbf + fp_rbf > 0 else 0

        accuracy_poly = ((tp_poly + tn_poly) / nObsTest) * 100
        precision_poly = (tp_poly / (tp_poly + fp_poly)) * 100 if tp_poly + fp_poly > 0 else 0

        recall_linear = (tp_linear / (tp_linear + fn_linear)) * 100 if tp_linear + fn_linear > 0 else 0
        recall_rbf = (tp_rbf / (tp_rbf + fn_rbf)) * 100 if tp_rbf + fn_rbf > 0 else 0
        recall_poly = (tp_poly / (tp_poly + fn_poly)) * 100 if tp_poly + fn_poly > 0 else 0

        f1_score_linear = (2 * (precision_linear * recall_linear) / (precision_linear + recall_linear)) if (precision_linear + recall_linear) != 0 else 0
        f1_score_rbf = (2 * (precision_rbf * recall_rbf) / (precision_rbf + recall_rbf)) if (precision_rbf + recall_rbf) != 0 else 0
        f1_score_poly = (2 * (precision_poly * recall_poly) / (precision_poly + recall_poly))  if (precision_poly + recall_poly) != 0 else 0

        results = {
            'Method': ['Linear', 'RBF', 'Poly'],
            'Number components': n_components,
            'TP': [tp_linear, tp_rbf, tp_poly],
            'FP': [fp_linear, fp_rbf, fp_poly],
            'TN': [tn_linear, tn_rbf, tn_poly],
            'FN': [fn_linear, fn_rbf, fn_poly],
            'Accuracy': [accuracy_linear, accuracy_rbf, accuracy_poly],
            'Precision': [precision_linear, precision_rbf, precision_poly],
            'Recall': [recall_linear, recall_rbf, recall_poly],
            'F1 Score': [f1_score_linear, f1_score_rbf, f1_score_poly],
            'Labels': [predicted_labels_linear, predicted_labels_rbf, predicted_labels_poly],
            'ConfusionMatrix': [
                confusion_matrix(actual_labels_linear, predicted_labels_linear),
                confusion_matrix(actual_labels_rbf, predicted_labels_rbf),
                confusion_matrix(actual_labels_poly, predicted_labels_poly)
            ]
        }
        all_results.append(results)

    # DataFrame from the results
    df = pd.concat([pd.DataFrame(res) for res in all_results], ignore_index=True)

    # DataFrame to an Excel file
    # df.to_excel(name_excel+'resultados_OneClassSVM_pca.xlsx', index=False)
    df.to_excel(os.path.join('resultados_script_dumb', f'{name_excel}_resultados_OneClassSVM_pca.xlsx'), index=False)


    # Find the index of the row with the best F1 score
    best_f1_index = df['F1 Score'].idxmax()
    best_f1_value = df['F1 Score'].max()

    best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']
    best_number_components=df.loc[best_f1_index,'Number components']
    best_kernel = df.loc[best_f1_index,'Method']
    best_labels = df.loc[best_f1_index,'Labels']

    # Plot the best confusion matrix if it exists
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['Normal', 'DNS TUNNEL'], yticklabels=['Normal', 'DNS TUNNEL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Best Confusion Matrix One Class SVM with pca: {best_number_components} and Kernel {best_kernel}')
    plt.show()

    print("F1 Score OSVM PCA: ", best_f1_value)

    return best_labels



from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

#########################################################Isolation_forest without pca###################################################
def isolation_forest(train_features, testFeatures_normal, testFeatures_dns, o3testClass, name_excel):

    i3Ctest = np.vstack((testFeatures_normal, testFeatures_dns))
    i3train = np.vstack((train_features))
    
    # Create an Isolation Forest instance
    isolation_forest = IsolationForest(contamination=0.1)

    # Fit the model on the training features
    isolation_forest.fit(i3train)

    # Predict anomalies on the test features
    anomaly_predictions = isolation_forest.predict(i3Ctest) #-1 anomalia, 1 comportamento normal

    # Convert predictions to binary (0 for normal, 2 for anomaly)
    binary_predictions = (anomaly_predictions == -1).astype(int) * 2

    nObsTest,nFea = i3Ctest.shape
    actual_labels = []

    for i in range(nObsTest):
        actual_labels.append(o3testClass[i][0])
    results = []
    # Calculate the confusion matrix
    confusion_matrix_result = confusion_matrix(actual_labels, binary_predictions)
    # Calculate precision, recall, and F1 score
    TN = confusion_matrix_result[0, 0]
    FP = confusion_matrix_result[0, 1]
    FN = confusion_matrix_result[1, 0]
    TP = confusion_matrix_result[1, 1]
    precision = precision_score(actual_labels, binary_predictions, pos_label=2)
    recall = recall_score(actual_labels, binary_predictions, pos_label=2)
    f1 = f1_score(actual_labels, binary_predictions, pos_label=2)
    #print(f1)

    results = {
            'TP': [TP],
            'FP': [FP],
            'TN': [TN],
            'FN': [FN],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1],
            'ConfusionMatrix': [
                confusion_matrix_result
            ]
        }    
        

    df = pd.DataFrame(results)

    # df.to_excel(name_excel+'resultados_SVM_PCA.xlsx', index=False)
    df.to_excel(os.path.join('resultados_script_dumb', f'{name_excel}_resultados_isolation_forest.xlsx'), index=False)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues', fmt='d',
                xticklabels=['Normal', 'DNS TUNNEL'], yticklabels=['Normal', 'DNS TUNNEL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f' Confusion Matrix Isolation Forest')
    plt.show()

    print("F1 Score Isolation Forest: ", f1)

    return binary_predictions



#########################################################Isolation_forest with pca###################################################

def isolation_forest_with_pca(train_features, testFeatures_normal, testFeatures_dns, o3testClass, name_excel):
    i3Ctest = np.vstack((testFeatures_normal, testFeatures_dns))
    i3train = np.vstack((train_features))

    all_results = []
    results=[]

    components_to_test = [5, 10, 15, 20, 25]

    for n_components in components_to_test:
        # pca_features = apply_pca(i3train, i3Ctest, n_components)
        actual_labels=[]
        binary_predictions=[]
        confusion_matrix_result=[]

        pca = PCA(n_components=n_components)
        i3train_pca = pca.fit_transform(i3train)
        i3Ctest_pca = pca.transform(i3Ctest)
        nObsTest, nFea = i3Ctest_pca.shape

        # Create an Isolation Forest instance
        isolation_forest = IsolationForest(contamination=0.1)

        # Fit the model on the training features
        isolation_forest.fit(i3train_pca)

        # Predict anomalies on the test features
        anomaly_predictions = isolation_forest.predict(i3Ctest_pca)

        for i in range(nObsTest):
            actual_labels.append(o3testClass[i][0])
            if(anomaly_predictions[i]==-1):
                binary_predictions.append(2.0)
            else:
                binary_predictions.append(0.0)

        # Calculate the confusion matrix
        confusion_matrix_result = confusion_matrix(actual_labels, binary_predictions)
        TN = confusion_matrix_result[0, 0]
        FP = confusion_matrix_result[0, 1]
        FN = confusion_matrix_result[1, 0]
        TP = confusion_matrix_result[1, 1]
        # Calculate precision, recall, and F1 score
        precision = precision_score(actual_labels, binary_predictions, pos_label=2)
        recall = recall_score(actual_labels, binary_predictions, pos_label=2)
        f1 = f1_score(actual_labels, binary_predictions, pos_label=2)

        # Store results in a dictionary
        results = {
            'TP': [TP],
            'FP': [FP],
            'TN': [TN],
            'FN': [FN],
            'Number components': n_components,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Labels': [binary_predictions],
            'ConfusionMatrix': [confusion_matrix_result]
        }

        all_results.append(results)

    df = pd.concat([pd.DataFrame(res) for res in all_results], ignore_index=True)


    # df.to_excel(name_excel+'resultados_SVM_PCA.xlsx', index=False)
    df.to_excel(os.path.join('resultados_script_dumb', f'{name_excel}_resultados_isolation_forest_pca.xlsx'), index=False)

    # Find the index of the row with the best F1 score
    best_f1_index = df['F1 Score'].idxmax()
    best_f1_value = df['F1 Score'].max()
    best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']
    best_number_components=df.loc[best_f1_index,'Number components']
    best_labels=df.loc[best_f1_index,'Labels']
        
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
                    xticklabels=['Normal', 'DNS TUNNEL'], yticklabels=['Normal', 'DNS TUNNEL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f' Confusion Matrix Isolation Forest with PCA: {best_number_components}')
    plt.show()

    print("F1 Score Isolation Forest pca: ", best_f1_value)

    return best_labels



##########################################################localOutflier without pca######################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
def lof_classification(train_features, test_features_normal, test_features_dns, o3testClass, name_excel):
    
    i3_train = np.vstack((train_features))
    i3_test = np.vstack((test_features_normal, test_features_dns))
    actual_labels=[]
    predictions=[]
    nObsTest, nFea = i3_test.shape

    # Create a Local Outlier Factor instance
    lof = LocalOutlierFactor(n_neighbors=80, contamination=0.5)

    # Fit the model on the training features
    lof.fit(i3_train)

    # Predict anomaly scores on the test features
    anomaly_predictions = lof.fit_predict(i3_test)
    #print(anomaly_predictions)
    for i in range(nObsTest):
        actual_labels.append(o3testClass[i][0])
        if(anomaly_predictions[i]==-1):
            predictions.append(2.0)
        else:
            predictions.append(0.0)
    #print(actual_labels)
    # Evaluate the performance
    confusion_matrix_result = confusion_matrix(actual_labels, predictions)
    precision = confusion_matrix_result[1, 1] / (confusion_matrix_result[1, 1] + confusion_matrix_result[0, 1]) if (
            confusion_matrix_result[1, 1] + confusion_matrix_result[0, 1]) > 0 else 0
    recall = confusion_matrix_result[1, 1] / (confusion_matrix_result[1, 1] + confusion_matrix_result[1, 0]) if (
            confusion_matrix_result[1, 1] + confusion_matrix_result[1, 0]) > 0 else 0
    f1 = f1_score(actual_labels, predictions,pos_label=2)
    results = {
        'TP': confusion_matrix_result[1, 1],
        'FP': confusion_matrix_result[0, 1],
        'TN': confusion_matrix_result[0, 0],
        'FN': confusion_matrix_result[1, 0],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ConfusionMatrix': [confusion_matrix_result]
    }

    df = pd.DataFrame(results)

    df.to_excel(os.path.join('resultados_script_dumb', f'{name_excel}_resultados_lof.xlsx'), index=False)

    # best_f1_value = df['F1 Score'].max()

    print("F1 Score localOutflier: ", f1)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues', fmt='d',
                xticklabels=['Normal', 'DNS TUNNEL'], yticklabels=['Normal', 'DNS TUNNEL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Local Outlier Factor with F1 score of {f1}')
    plt.show()

    return predictions


##########################################################localOutflier with pca######################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
    
def lof_classification_with_pca(train_features, test_features_normal, test_features_dns, o3testClass, name_excel):
    
    components_to_test = [5, 10, 15, 20, 25]
    i3_train = np.vstack((train_features))
    i3_test = np.vstack((test_features_normal, test_features_dns))
    actual_labels = []
    predictions = []
    nObsTest, nFea = i3_test.shape

    best_f1_score = 0
    best_confusion_matrix = None
    best_pca_components = 0

    for n_components in components_to_test:
        # Apply PCA
        pca = PCA(n_components=n_components)
        i3_train_pca = pca.fit_transform(i3_train)
        i3_test_pca = pca.transform(i3_test)

        # Create a Local Outlier Factor instance
        lof = LocalOutlierFactor(n_neighbors=80, contamination=0.5)

        # Fit the model on the training features
        lof.fit(i3_train_pca)

        # Predict anomaly scores on the test features
        anomaly_predictions = lof.fit_predict(i3_test_pca)

        for i in range(nObsTest):
            actual_labels.append(o3testClass[i][0])
            if anomaly_predictions[i] == -1:
                predictions.append(2.0)
            else:
                predictions.append(0.0)

        # Evaluate the performance
        confusion_matrix_result = confusion_matrix(actual_labels, predictions)
        f1 = f1_score(actual_labels, predictions, pos_label=2)

        if f1 > best_f1_score:
            best_f1_score = f1
            best_confusion_matrix = confusion_matrix_result
            best_pca_components = n_components
            best_labels = predictions

        # Reset lists for the next iteration
        actual_labels = []
        predictions = []

    # Print and save results for the best PCA components
    results = {
        'TP': best_confusion_matrix[1, 1],
        'FP': best_confusion_matrix[0, 1],
        'TN': best_confusion_matrix[0, 0],
        'FN': best_confusion_matrix[1, 0],
        'Precision': best_confusion_matrix[1, 1] / (best_confusion_matrix[1, 1] + best_confusion_matrix[0, 1]) if (
                best_confusion_matrix[1, 1] + best_confusion_matrix[0, 1]) > 0 else 0,
        'Recall': best_confusion_matrix[1, 1] / (best_confusion_matrix[1, 1] + best_confusion_matrix[1, 0]) if (
                best_confusion_matrix[1, 1] + best_confusion_matrix[1, 0]) > 0 else 0,
        'F1 Score': best_f1_score,
        'Lables': [best_labels],
        'ConfusionMatrix': [best_confusion_matrix]
    }

    df = pd.DataFrame(results)

    df.to_excel(os.path.join('resultados_script_dumb', f'{name_excel}_resultados_lof_pca.xlsx'), index=False)

    best_f1_value = df['F1 Score'].max()

    print("F1 Score localOutflier with pca: ", best_f1_value)
    
    # Plot the confusion matrix for the best PCA components
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['Normal', 'DNS TUNNEL'], yticklabels=['Normal', 'DNS TUNNEL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix Local Outlier Factor with {best_pca_components} PCA components')
    plt.show()

    return best_labels

from sklearn.mixture import GaussianMixture


#############################################################Gaussian mixture model#############################################
def gmm_classification(train_features, test_features_normal, test_features_dns, o3_test_class, name_excel, n_components=2):
    train_data = np.vstack((train_features))
    test_data = np.vstack((test_features_normal, test_features_dns))
    nObsTest, nFea = test_data.shape
    actual_labels=[]
    predictions=[]

    # Create a Gaussian Mixture Model instance
    gmm = GaussianMixture(n_components=2, random_state=42)

    # Fit the model on the training features
    gmm.fit(train_data)

    # Predict cluster assignments and anomaly scores on the test features
    cluster_assignments = gmm.predict(test_data)
    anomaly_scores = -gmm.score_samples(test_data)
    threshold = np.percentile(anomaly_scores, 40)  # Consider top 60% as anomalies

    # Predict anomalies based on the threshold
    anomaly_predictions = (anomaly_scores > threshold).astype(int)
    #print(anomaly_predictions)
    for i in range(nObsTest):
        actual_labels.append(o3_test_class[i][0])
        if(anomaly_predictions[i]==1):
            predictions.append(2.0)
        else:
            predictions.append(0.0)
    

    # Evaluate the performance
    confusion_matrix_result = confusion_matrix(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions, pos_label=2,average='macro')
    recall = recall_score(actual_labels, predictions,pos_label=2, average='macro')
    f1 = f1_score(actual_labels, predictions,pos_label=2, average='macro')

    results = {
        'TP': confusion_matrix_result[1, 1],
        'FP': confusion_matrix_result[0, 1],
        'TN': confusion_matrix_result[0, 0],
        'FN': confusion_matrix_result[1, 0],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ConfusionMatrix': [confusion_matrix_result]
    }

    df = pd.DataFrame(results)

    df.to_excel(os.path.join('resultados_script_dumb', f'{name_excel}_resultados_gmm.xlsx'), index=False)
    
    best_f1_value = df['F1 Score'].max()

    print("F1 Score Gaussian mixture model: ", best_f1_value)
    

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues', fmt='d',
                xticklabels=['Normal', 'DNS TUNNEL'], yticklabels=['Normal', 'DNS TUNNEL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Gaussian Mixture Model')
    plt.show()

    return anomaly_predictions


def gmm_classification_with_pca(train_features, test_features_normal, test_features_dns, o3_test_class, name_excel):
    train_data = np.vstack((train_features))
    test_data = np.vstack((test_features_normal, test_features_dns))
    nObsTest, nFea = test_data.shape
    components_to_test = [5, 10, 15, 20, 25]
    best_f1_score = 0
    best_pca_component = 0
    best_confusion_matrix = None
    actual_labels = []
    predictions = []

    for n_components in components_to_test:
        # Apply PCA
        pca = PCA(n_components=n_components)
        train_data_pca = pca.fit_transform(train_data)
        test_data_pca = pca.transform(test_data)

        # Create a Gaussian Mixture Model instance
        gmm = GaussianMixture(n_components=2, random_state=42)

        # Fit the model on the training features
        gmm.fit(train_data_pca)

        # Predict cluster assignments and anomaly scores on the test features
        cluster_assignments = gmm.predict(test_data_pca)
        anomaly_scores = -gmm.score_samples(test_data_pca)
        threshold = np.percentile(anomaly_scores, 40)  # Consider top 60% as anomalies

        # Predict anomalies based on the threshold
        anomaly_predictions = (anomaly_scores > threshold).astype(int)
        for i in range(nObsTest):
            actual_labels.append(o3_test_class[i][0])
            if(anomaly_predictions[i]==1):
                predictions.append(2.0)
            else:
                predictions.append(0.0)

        # Evaluate the performance
        confusion_matrix_result = confusion_matrix(actual_labels, predictions)
        precision = precision_score(actual_labels, predictions, pos_label=2, average='macro')
        recall = recall_score(o3_test_class, predictions, pos_label=2, average='macro')
        f1 = f1_score(actual_labels, predictions, pos_label=2, average='macro')

        if f1 > best_f1_score:
            best_f1_score = f1
            best_pca_component = n_components
            best_confusion_matrix = confusion_matrix_result
            best_labels = anomaly_predictions
            precision,recall=0
            
        predictions = []
        actual_labels = []

    results = {
        'TP': best_confusion_matrix[1, 1],
        'FP': best_confusion_matrix[0, 1],
        'TN': best_confusion_matrix[0, 0],
        'FN': best_confusion_matrix[1, 0],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Lables': [best_labels],
        'ConfusionMatrix': [best_confusion_matrix]
    }

    df = pd.DataFrame(results)

    df.to_excel(os.path.join('resultados_script_dumb', f'{name_excel}_resultados_gmm_pca_{n_components}.xlsx'), index=False)

    print("Best F1 Score with PCA: ", best_f1_score)
    #print("Best PCA Component: ", best_pca_component)

    # Plot the confusion matrix for the best PCA components
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['Normal', 'DNS TUNNEL'], yticklabels=['Normal', 'DNS TUNNEL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix Gaussian mixture model with {best_pca_component} PCA components')
    plt.show()

    return best_labels

########### Main Code #############

## -- 3 -- ##

# features: É a matriz de características (ou atributos) onde cada linha representa uma observação e cada coluna representa um atributo específico.
# oClass: É a matriz de classes correspondentes a cada observação features.

Classes={0:'Bruno',1:'Marta',2:'DNS'}
plt.ion()
nfig=1

## -- 2 -- ##
features_bruno=np.loadtxt("features_bruno.dat")
features_marta=np.loadtxt("features_marta.dat")
features_dns=np.loadtxt("features_dns_tunneling_smart.dat")


#It assigns class labels (0 for Bruno and  Marta, and 2 for dns_tunneling) to the respective datasets
#cada classe vai conter:mean, median and standard deviation  and also the silence periods features(mean median and deviation) and percentis for upload and download 
oClass_bruno=np.ones((len(features_bruno),1))*0
oClass_marta=np.ones((len(features_marta),1))*0
oClass_dns=np.ones((len(features_dns),1))*2


#resulta num conjunto de features que contém todos os dados dessas diferentes fontes combinados verticalmente.
#features=np.vstack((features_marta,features_bruno,features_dns))
#um único array oClass que contém todas as classes correspondentes às observações do conjunto de dados combinado features.
#oClass=np.vstack((oClass_marta,oClass_bruno,oClass_dns))

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
#pB, pM: Calculam o tamanho do conjunto de treino para cada categoria com base na percentagem definida.
pB=int(len(features_bruno)*percentage)
trainFeatures_bruno=features_bruno[:pB,:]
pM=int(len(features_marta)*percentage)
trainFeatures_marta=features_marta[:pM,:]
# pD=int(len(features_dns)*percentage)
# trainFeatures_dns=features_dns[:pD,:]

#i2train: Build train features of normal behavior
# i2train=np.vstack((trainFeatures_bruno,trainFeatures_marta))
# o2trainClass=np.vstack((oClass_bruno[:pB],oClass_marta[:pM]))

#:ii
# i3Ctrain=np.vstack((trainFeatures_bruno,trainFeatures_marta,trainFeatures_dns))
# o3trainClass=np.vstack((oClass_bruno[:pB],oClass_marta[:pM],oClass_dns[:pD]))

#:iii
testFeatures_bruno=features_bruno[pB:,:]
testFeatures_marta=features_marta[pM:,:]
testFeatures_dns=features_dns

#----------------------------------------------------------Testing Bruno Behaviour----------------------------------------------
name_excel="bruno_smart"

o3testClass=np.vstack((oClass_bruno[pB:],oClass_dns))
o3trainClass=np.vstack((oClass_bruno[:pB]))

# labels1 = one_class_svm(trainFeatures_bruno, testFeatures_bruno, testFeatures_dns, o3testClass,name_excel)
# labels2 = one_class_svm_with_pca(trainFeatures_bruno, testFeatures_bruno, testFeatures_dns, o3testClass,name_excel)
# labels3 = isolation_forest(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass, name_excel)
# labels4 = isolation_forest_with_pca(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
# labels5 = lof_classification(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
# labels6 = lof_classification_with_pca(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)

# final_lables = [labels1, labels2, labels3, labels4, labels5, labels6]
# final_lables = [labels1, labels3, labels4, labels5, labels6]
#labels1 = one_class_svm(trainFeatures_bruno, testFeatures_bruno, testFeatures_dns, o3testClass,name_excel)
# labels2 = one_class_svm_with_pca(trainFeatures_bruno, testFeatures_bruno, testFeatures_dns, o3testClass,name_excel)
#labels3 = isolation_forest(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass, name_excel)
#labels4 = isolation_forest_with_pca(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
#labels5 = lof_classification(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
#labels6 = lof_classification_with_pca(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
#labels7=gmm_classification(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
#labels8=gmm_classification_with_pca(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
# final_lables = [labels1, labels2, labels3, labels4, labels5, labels6]
#final_lables = [labels1, labels3, labels4, labels5, labels6]

# print("\nFinal Labels:\n")
# for i in range(len(labels1)):
#     for j in range(len(final_lables)):
#         print(final_lables[j][i])

#     print("\n")


#nao estao a funcionar bem
#random_forest_classification_without_pca(trainFeatures_bruno, testFeatures_bruno, testFeatures_dns, o3trainClass, o3testClass, name_excel)
# linear_regression_model(trainFeatures_bruno, testFeatures_bruno, testFeatures_dns, o3trainClass, o3testClass, name_excel)

#----------------------------------------------------------Testing Marta Behaviour----------------------------------------------
name_excel="marta_smart"

o3testClass=np.vstack((oClass_marta[:pM],oClass_dns))
o3trainClass=np.vstack(oClass_marta[:pM])

# one_class_svm(trainFeatures_marta, testFeatures_marta, testFeatures_dns, o3testClass,name_excel)
# one_class_svm_with_pca(trainFeatures_marta, testFeatures_marta, testFeatures_dns, o3testClass,name_excel)
# isolation_forest(trainFeatures_marta,testFeatures_marta, testFeatures_dns,o3testClass)
# isolation_forest_with_pca(trainFeatures_marta,testFeatures_marta, testFeatures_dns,o3testClass,name_excel)

# one_class_svm(trainFeatures_marta, testFeatures_marta, testFeatures_dns, o3testClass,name_excel)
# one_class_svm_with_pca(trainFeatures_marta, testFeatures_marta, testFeatures_dns, o3testClass,name_excel)
# neural_network_classification(trainFeatures_marta, testFeatures_marta, trainFeatures_dns, testFeatures_dns, o3trainClass, o3testClass,name_excel)
# neural_network_classification_with_pca(trainFeatures_marta, testFeatures_marta, trainFeatures_dns, testFeatures_dns, o3trainClass, o3testClass,name_excel)

# # # Wait for user input before exiting
# waitforEnter(fstop=True)
