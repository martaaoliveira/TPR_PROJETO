import os
import numpy as np
import pandas as pd
from profiling import waitforEnter, one_class_svm, one_class_svm_with_pca, isolation_forest, isolation_forest_with_pca, lof_classification, lof_classification_with_pca,gmm_classification,gmm_classification_with_pca
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")

# def get_excel_files(directory):
#     excel_files = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".xlsx") or filename.endswith(".xls"):
#             file_path = os.path.join(directory, filename)
#             excel_files.append(file_path)
#     return excel_files


# def get_metrics(df):
#     TN = df['TN'].sum()
#     FP = df['FP'].sum()
#     FN = df['FN'].sum()
#     TP = df['TP'].sum()
#     return TN, FP, FN, TP

def main():
    features_bruno=np.loadtxt("features_bruno.dat")
    features_marta=np.loadtxt("features_marta.dat")
    features_dns=np.loadtxt("features_dns_tunneling_smart.dat")

    oClass_bruno=np.ones((len(features_bruno),1))*0
    oClass_marta=np.ones((len(features_marta),1))*0
    oClass_dns=np.ones((len(features_dns),1))*2

    percentage=0.5
    #pB, pM: Calculam o tamanho do conjunto de treino para cada categoria com base na percentagem definida.
    pB=int(len(features_bruno)*percentage)
    trainFeatures_bruno=features_bruno[:pB,:]
    pM=int(len(features_marta)*percentage)
    trainFeatures_marta=features_marta[:pM,:]

    testFeatures_bruno=features_bruno[pB:,:]
    testFeatures_marta=features_marta[pM:,:]
    testFeatures_dns=features_dns


    #--------------------- Bruno's Model ------------------------
    name_excel="bruno_smart"

    o3testClass=np.vstack((oClass_bruno[pB:],oClass_dns))
    o3trainClass=np.vstack((oClass_bruno[:pB]))
    i3_test = np.vstack((testFeatures_bruno, testFeatures_dns))
    nObsTest, nFea = i3_test.shape
    actual_labels = []
    for i in range(nObsTest):
        actual_labels.append(o3testClass[i][0])

    labels1 = one_class_svm(trainFeatures_bruno, testFeatures_bruno, testFeatures_dns, o3testClass,name_excel)
    # labels2 = one_class_svm_with_pca(trainFeatures_bruno, testFeatures_bruno, testFeatures_dns, o3testClass,name_excel)
    labels3 = isolation_forest(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass, name_excel)
    labels4 = isolation_forest_with_pca(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
    labels5 = lof_classification(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
    labels6 = lof_classification_with_pca(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)

    #--------------------- Marta's Model ------------------------

    # name_excel="marta_smart"

    # o3testClass=np.vstack((oClass_marta[pB:],oClass_dns))
    # o3trainClass=np.vstack((oClass_marta[:pB]))
    # i3_test = np.vstack((testFeatures_marta, testFeatures_dns))
    # nObsTest, nFea = i3_test.shape
    # actual_labels = []
    # for i in range(nObsTest):
    #     actual_labels.append(o3testClass[i][0])

    # labels1 = one_class_svm(trainFeatures_marta, testFeatures_marta, testFeatures_dns, o3testClass,name_excel)
    # # labels2 = one_class_svm_with_pca(trainFeatures_marta, testFeatures_marta, testFeatures_dns, o3testClass,name_excel)
    # labels3 = isolation_forest(trainFeatures_marta,testFeatures_marta, testFeatures_dns,o3testClass, name_excel)
    # labels4 = isolation_forest_with_pca(trainFeatures_marta,testFeatures_marta, testFeatures_dns,o3testClass,name_excel)
    # labels5 = lof_classification(trainFeatures_marta,testFeatures_marta, testFeatures_dns,o3testClass,name_excel)
    # labels6 = lof_classification_with_pca(trainFeatures_marta,testFeatures_marta, testFeatures_dns,o3testClass,name_excel)


    # lables_list = [labels1, labels2, labels3, labels4, labels5, labels6]
    lables_list = [labels1, labels3, labels4, labels5, labels6]
=======
    #labels1 = one_class_svm(trainFeatures_bruno, testFeatures_bruno, testFeatures_dns, o3testClass,name_excel)
    #labels2 = one_class_svm_with_pca(trainFeatures_bruno, testFeatures_bruno, testFeatures_dns, o3testClass,name_excel)
    #labels3 = isolation_forest(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass, name_excel)
    #labels4 = isolation_forest_with_pca(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
    #labels5 = lof_classification(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
    #labels6 = lof_classification_with_pca(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
    labels7=gmm_classification(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
    labels8=gmm_classification_with_pca(trainFeatures_bruno,testFeatures_bruno, testFeatures_dns,o3testClass,name_excel)
    lables_list = [labels7, labels8]
    #lables_list = [labels1, labels2,labels3, labels4, labels5,labels7,labels8]
    final_lables = []

    for i in range(len(labels7)):
        cnt = 0
        for j in range(len(lables_list)):
            if (lables_list[j][i] == 0):
                cnt += 1
        
        if (cnt > len(lables_list) / 2):
            final_lables.append(0)
        else: 
            final_lables.append(2)


    confusion_matrix_result = confusion_matrix(actual_labels, final_lables)
    f1 = f1_score(actual_labels, final_lables, pos_label=2)
    f1=round(f1, 2)
    print("F1 Score Ensemble: ", f1)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues', fmt='d',
                xticklabels=['Normal', 'DNS TUNNEL'], yticklabels=['Normal', 'DNS TUNNEL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix Local Ensemble with F1 score of {f1}')
    plt.show()

    # # # Wait for user input before exiting
    waitforEnter(fstop=True)

if __name__ == "__main__":
    main()
