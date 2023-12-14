import argparse
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os



#adicionar features extras
# (média, mediana, desvio padrão, "silêncio" e "atividade") em um único array
# O axis=0 especifica o calculo dessas estatísticas ao longo das colunas (axis 0) do array. 
# Isso significa que essas estatísticas serão calculadas para cada métrica, criando um array de estatísticas descritivas com o mesmo número de colunas que o array data.

def extractStatsAdv(data,threshold=0):
    nSamp=data.shape
    print(data)

    MX= np.max(data,axis=0)
    M1=np.mean(data,axis=0)
    Md1=np.median(data,axis=0) #axis=0 coluna a coluna
    Std1=np.std(data,axis=0)
    p=[75,90,95,98]
    Pr1=np.array(np.percentile(data,p,axis=0))

    silence_faux = np.zeros(3)
    activity_features = np.zeros(3)

    silence,activity=extratctSilenceActivity(data,threshold)
    
    #Se silence não estiver vazio, ou seja, se houver períodos de silêncio nos dados, o código dentro deste bloco será executado.
    if len(silence)>0:
        #três estatísticas relacionadas aos períodos de silêncio:
        silence_faux=np.array([len(silence),np.mean(silence),np.std(silence)])
    #
    else:
        silence_faux=np.zeros(3)
    
    # if len(activity)>0:
    #     activity_faux=np.array([len(activity),np.mean(activity),np.std(activity)])
    # else:
    #     activity_faux=np.zeros(3)
    # activity_features=np.hstack((activity_features,activity_faux))  
    
    features=np.hstack((MX,M1,Md1,Std1,silence_faux,Pr1))
    return(features)

def extratctSilenceActivity(data,threshold=0):
    if(data[0]<=threshold):
        s=[1]
        a=[]
    else:
        s=[]
        a=[1]
    for i in range(1,len(data)):
        if(data[i-1]>threshold and data[i]<=threshold):
            s.append(1)
        elif(data[i-1]<=threshold and data[i]>threshold):
            a.append(1)
        elif (data[i-1]<=threshold and data[i]<=threshold):
            s[-1]+=1
        else:
            a[-1]+=1
    return(s,a)



# obsFeatures: Este é um array que armazena as características calculadas para uma única janela de observação. 
# A cada iteração do loop, um novo conjunto de características é calculado para a janela atual, e essas características são armazenadas em obsFeatures. 
# No final da iteração, obsFeatures conterá todas as características calculadas para essa janela.


# wmFeatures: Este é um array que armazena as características calculadas para uma única métrica (coluna) dos dados originais dentro da janela de observação. 
# O loop for aninhado percorre todas as métricas (colunas) dos dados originais, e a cada iteração, um novo conjunto de características é calculado para a métrica atual dentro da janela de observação. 
# O resultado das características para cada métrica é armazenado em wmFeatures, e no final do loop interno, wmFeatures conterá todas as características calculadas para todas as métricas dentro da janela atual.

# A função np.hstack é usada para empilhar horizontalmente os arrays wmFeatures em obsFeatures. 
# Isso é feito para combinar as características de todas as métricas em uma única matriz


#nMetrics é o número de colunas (ou métricas) em seus dados.

# 4 métricas originais(do ficheiro OutFile) x 3 estatísticas (média, mediana, desvio padrão) = 12 elementos no array
#se adicionarmos os percentis fica  4 métricas originais(do ficheiro OutFile) x 7 estatísticas (média, mediana, desvio padrão, 4 percentis)= 28 elementos no array
#cada "array" de 28 elementos representa as features de 1 janela 

#Opcao argumento "1"
def seqObsWindow(data,lengthObsWindow):
    iobs=0
    #Nmetrics= tamanho da janela
    nSamples,nMetrics=data.shape
    while iobs*lengthObsWindow<nSamples-lengthObsWindow:
        obsFeatures=np.array([])
        for m in np.arange(nMetrics):
            wmFeatures=extractStatsAdv(data[iobs*lengthObsWindow:(iobs+1)*lengthObsWindow,m])
            obsFeatures=np.hstack((obsFeatures,wmFeatures))
        iobs+=1
        
        if 'allFeatures' not in locals():
            allFeatures=obsFeatures.copy()
        else:
            allFeatures=np.vstack((allFeatures,obsFeatures))
    return(allFeatures)

#Opcao argumento "2"
# Esta a usar extractStatsADV (contem atividades de silêncio) 
def slidingObsWindow(data,lengthObsWindow,slidingValue):
    iobs=0
    nSamples,nMetrics=data.shape
    while iobs*slidingValue<nSamples-lengthObsWindow:
        obsFeatures=np.array([])
        for m in np.arange(nMetrics):
            #wmFeatures=extractStats(data[iobs*slidingValue:iobs*slidingValue+lengthObsWindow,m])
            wmFeatures = extractStatsAdv(data[iobs * slidingValue:iobs * slidingValue + lengthObsWindow, m])
            obsFeatures=np.hstack((obsFeatures,wmFeatures))
        iobs+=1
        if 'allFeatures' not in locals():
            allFeatures=obsFeatures.copy()
        else:
            allFeatures=np.vstack((allFeatures,obsFeatures))
    return(allFeatures)

#Opcao argumento "3"
def slidingMultObsWindow(data,allLengthsObsWindow,slidingValue):
    iobs=0
    nSamples,nMetrics=data.shape
    while iobs*slidingValue<nSamples-max(allLengthsObsWindow):
        obsFeatures=np.array([])
        for lengthObsWindow in allLengthsObsWindow:
            for m in np.arange(nMetrics):
                wmFeatures=extractStatsAdv(data[iobs*slidingValue:iobs*slidingValue+lengthObsWindow,m])
                obsFeatures=np.hstack((obsFeatures,wmFeatures))
            iobs+=1
        
        if 'allFeatures' not in locals():
            allFeatures=obsFeatures.copy()
        else:
            allFeatures=np.vstack((allFeatures,obsFeatures))
    return(allFeatures)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?',required=True, help='input file')
    parser.add_argument('-m', '--method', nargs='?',required=False, help='obs. window creation method',default=2)
    parser.add_argument('-w', '--widths', nargs='*',required=False, help='list of observation windows widths',default=60)
    parser.add_argument('-s', '--slide', nargs='?',required=False, help='observation windows slide value',default=0)
    args=parser.parse_args()
    
    fileInput=args.input
    method=int(args.method)
    lengthObsWindow=[int(w) for w in args.widths]
    slidingValue=int(args.slide)
        
    data=np.loadtxt(fileInput,dtype=int)
    if method==1:
        fname=''.join(fileInput.split('.')[:-1])+"_features_m{}_w{}".format(method,lengthObsWindow)
    else:
        fname=''.join(fileInput.split('.')[:-1])+"_features_m{}_w{}_s{}".format(method,lengthObsWindow,slidingValue)
    
    if method==1:
        print("\n\n### SEQUENTIAL Observation Windows with Length {} ###".format(lengthObsWindow[0]))
        features=seqObsWindow(data,lengthObsWindow[0])
        print(features)
        print(fname)
        np.savetxt(fname,features,fmt='%d')
    elif method==2:
        print("\n\n### SLIDING Observation Windows with Length {} and Sliding {} ###".format(lengthObsWindow[0],slidingValue))
        features=slidingObsWindow(data,lengthObsWindow[0],slidingValue)
        print(features)
        print(fname)
        np.savetxt(fname,features,fmt='%d')
    elif method==3:
        print("\n\n### SLIDING Observation Windows with Lengths {} and Sliding {} ###".format(lengthObsWindow,slidingValue))    
        features=slidingMultObsWindow(data,lengthObsWindow,slidingValue)
        print(features)
        print(fname)
        np.savetxt(fname,features,fmt='%d')
            
        

if __name__ == '__main__':
    main()
