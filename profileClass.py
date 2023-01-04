import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import sys
import warnings
warnings.filterwarnings('ignore')


def printResults(truePositive, trueNegative, falsePositive, falseNegative):
    print("True Positives: {}, True Negatives: {}".format(truePositive, trueNegative))
    print("False Positives: {}, False Negatives: {}".format(falsePositive, falseNegative))
    print("Accuracy: {}%".format(((truePositive+trueNegative)/(truePositive+trueNegative+falsePositive+falseNegative))*100))
    precision = truePositive/(truePositive+falsePositive)
    print("Precision: {}%".format((precision)*100))
    recall = (truePositive)/(truePositive+falseNegative)
    print("Recall: {}%".format((recall)*100))
    print("F1-Score: {}".format(((2*(recall*precision))/(recall+precision))))


def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")
            
## -- 4 -- ##
def plotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r'] # adicionar cores caso tenha mais que 3 clientes
    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

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
    return(np.sqrt(np.sum(np.square(p-c))))

########### Main Code #############
Classes={0:'YouTube',1:'Browsing',2:'Exfiltration'}
#plt.ion()
nfig=1

## -- 3 -- ## done
features_browsing=np.loadtxt("output_browsing_abola_obs_features.dat")
features_yt=np.loadtxt("output_yt_1_obs_features.dat")
features_big=np.loadtxt("anomalia_obs_features.dat")

oClass_browsing=np.ones((len(features_browsing),1))*0
oClass_yt=np.ones((len(features_yt),1))*1
oClass_big=np.ones((len(features_big),1))*2


features=np.vstack((features_yt,features_browsing,features_big))
oClass=np.vstack((oClass_yt,oClass_browsing,oClass_big))

print('Train Stats Features Size:',features.shape)

## -- 4 -- ##
plt.figure(4)
plotFeatures(features,oClass,0,1)#0,8

## -- 5 -- ## 
features_browsingS=np.loadtxt("output_browsing_obs_sil_features.dat")
features_ytS=np.loadtxt("output_yt_obs_sil_features.dat")
features_bigS=np.loadtxt("Anomalia_obs_sil_features.dat")

featuresS=np.vstack((features_ytS,features_browsingS,features_bigS))
oClass=np.vstack((oClass_yt,oClass_browsing,oClass_big))

print('Train Silence Features Size:',featuresS.shape)
plt.figure(5)
plotFeatures(featuresS,oClass,0,2)


## -- 7 -- ## 
features_browsingW=np.loadtxt("output_browsing_abola_obs_per_features.dat")
features_ytW=np.loadtxt("output_yt_1_obs_per_features.dat")
features_bigW=np.loadtxt("anomalia_obs_per_features.dat")


featuresW=np.vstack((features_ytW,features_browsingW,features_bigW))
oClass=np.vstack((oClass_yt,oClass_browsing,oClass_big))

print('Train Wavelet Features Size:',featuresW.shape)
plt.figure(7)
plotFeatures(featuresW,oClass,3,6)


## -- 8 -- ##
#:1
percentage=0.5
pB=int(len(features_browsing)*percentage)
trainFeatures_browsing=features_browsing[:pB,:]
pYT=int(len(features_yt)*percentage)
trainFeatures_yt=features_yt[:pYT,:]

trainFeatures=np.vstack((trainFeatures_browsing,trainFeatures_yt))

trainFeatures_browsingS=features_browsingS[:pB,:]
trainFeatures_ytS=features_ytS[:pYT,:]

trainFeaturesS=np.vstack((trainFeatures_browsingS,trainFeatures_ytS))

trainFeatures_browsingW=features_browsingW[:pB,:]
trainFeatures_ytW=features_ytW[:pYT,:]

trainFeaturesW=np.vstack((trainFeatures_browsingW,trainFeatures_ytW))

o2trainClass=np.vstack((oClass_browsing[:pB],oClass_yt[:pYT]))
#i2trainFeatures=np.hstack((trainFeatures,trainFeaturesS,trainFeaturesW))
#i2trainFeatures=np.hstack((trainFeatures,trainFeaturesS))
i2trainFeatures=trainFeatures


#:3
pBig=int(len(features_big)*percentage)

testFeatures_browsing=features_browsing[pB:,:]
testFeatures_yt=features_yt[pYT:,:]
testFeatures_big=features_big[pBig:,:]

testFeaturesB=np.vstack((testFeatures_big))

testFeatures=np.vstack((testFeatures_browsing,testFeatures_yt))

testFeatures_browsingS=features_browsingS[pB:,:]
testFeatures_ytS=features_ytS[pYT:,:]
testFeatures_bigS=features_bigS[pBig:,:]

testFeaturesBS=np.vstack((testFeatures_bigS))
testFeaturesS=np.vstack((testFeatures_browsingS,testFeatures_ytS))

testFeatures_browsingW=features_browsingW[pB:,:]
testFeatures_ytW=features_ytW[pYT:,:]
testFeatures_bigW=features_bigW[pBig:,:]

testFeaturesBW=np.vstack((testFeatures_bigW))
testFeaturesW=np.vstack((testFeatures_browsingW,testFeatures_ytW))

o3testClass=np.vstack((oClass_browsing[pB:],oClass_yt[pYT:],oClass_big[pBig:]))
#i3testFeatures=np.hstack((testFeatures,testFeaturesS,testFeaturesW))
#i3testFeatures=np.hstack((testFeatures,testFeaturesS))
i3AtestFeatures=testFeaturesB
i3testFeatures=testFeatures


## -- 9 -- ##
from sklearn.preprocessing import MaxAbsScaler

i2trainScaler = MaxAbsScaler().fit(i2trainFeatures)
i2trainFeaturesN=i2trainScaler.transform(i2trainFeatures)

i3AtestFeaturesN=i2trainScaler.transform(i3AtestFeatures)
i3testFeaturesN=i2trainScaler.transform(i3testFeatures)
print("Mean values: ")
print(np.mean(i2trainFeaturesN,axis=0))
print("Standard deviation values: ")
print(np.std(i2trainFeaturesN,axis=0))

## -- 10 -- ## --------
from sklearn.decomposition import PCA

#experimentar varios valores para o n_componentes
pca = PCA(n_components=5, svd_solver='full')

i2trainPCA=pca.fit(i2trainFeaturesN)
i2trainFeaturesNPCA = i2trainPCA.transform(i2trainFeaturesN)

i3AtestFeaturesNPCA = i2trainPCA.transform(i3AtestFeaturesN)
i3testFeaturesNPCA = i2trainPCA.transform(i3testFeaturesN)

print(i2trainFeaturesNPCA.shape,o2trainClass.shape)
plt.figure(8)
plotFeatures(i2trainFeaturesNPCA,o2trainClass,0,1)


## -- 11 -- ##
from sklearn.preprocessing import MaxAbsScaler
centroids={}
for c in range(2):  # Only the first two classes
    pClass=(o2trainClass==c).flatten()
    centroids.update({c:np.mean(i2trainFeaturesN[pClass,:],axis=0)})
#print('All Features Centroids:\n',centroids)

truePositive=0
trueNegative=0
falsePositive=0
falseNegative=0

AnomalyThreshold=5
print('\n-- Anomaly Detection based on Centroids Distances --')
nObsTest,nFea=i3AtestFeaturesN.shape # ANOMALY
for i in range(nObsTest):
    x=i3AtestFeaturesN[i]
    dists=[distance(x,centroids[0]),distance(x,centroids[1])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        truePositive += 1
    else:
        result="OK"
        falseNegative += 1
       
nObsTest,nFea=i3testFeaturesN.shape # NORMAL
for i in range(nObsTest):
    x=i3testFeaturesN[i]
    dists=[distance(x,centroids[0]),distance(x,centroids[1])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        falsePositive += 1
    else:
        result="OK"
        trueNegative += 1
    #print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*dists,result))

printResults(truePositive, trueNegative, falsePositive, falseNegative)

## -- 12 -- ##

truePositive=0
trueNegative=0
falsePositive=0
falseNegative=0

centroids={}
for c in range(2):  # Only the first two classes
    pClass=(o2trainClass==c).flatten()
    centroids.update({c:np.mean(i2trainFeaturesNPCA[pClass,:],axis=0)})
#print('All Features Centroids:\n',centroids)

AnomalyThreshold=5
print('\n-- Anomaly Detection based on Centroids Distances (PCA Features) --')
nObsTest,nFea=i3AtestFeaturesNPCA.shape
for i in range(nObsTest):
    x=i3AtestFeaturesNPCA[i]
    dists=[distance(x,centroids[0]),distance(x,centroids[1])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        truePositive += 1
    else:
        result="OK"
        falseNegative += 1
       
nObsTest,nFea=i3testFeaturesNPCA.shape # NORMAL
for i in range(nObsTest):
    x=i3testFeaturesNPCA[i]
    dists=[distance(x,centroids[0]),distance(x,centroids[1])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        falsePositive += 1
    else:
        result="OK"
        trueNegative += 1
    #print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*dists,result))
printResults(truePositive, trueNegative, falsePositive, falseNegative)
    

## -- 13 -- ##
truePositive=0
trueNegative=0
falsePositive=0
falseNegative=0


from scipy.stats import multivariate_normal
print('\n-- Anomaly Detection based Multivariate PDF (PCA Features) --')
means={}
for c in range(2):
    pClass=(o2trainClass==c).flatten()
    means.update({c:np.mean(i2trainFeaturesNPCA[pClass,:],axis=0)})
#print(means)

covs={}
for c in range(2):
    pClass=(o2trainClass==c).flatten()
    covs.update({c:np.cov(i2trainFeaturesNPCA[pClass,:],rowvar=0)})
#print(covs)

AnomalyThreshold=0.0001
nObsTest,nFea=i3AtestFeaturesNPCA.shape
for i in range(nObsTest):
    x=i3AtestFeaturesNPCA[i,:]
    probs=np.array([multivariate_normal.pdf(x,means[0],covs[0]),multivariate_normal.pdf(x,means[1],covs[1])])
    if max(probs)<AnomalyThreshold:
        result="Anomaly"
        truePositive += 1
    else:
        result="OK"
        falseNegative += 1

nObsTest,nFea=i3testFeaturesNPCA.shape
for i in range(nObsTest):
    x=i3testFeaturesNPCA[i,:]
    probs=np.array([multivariate_normal.pdf(x,means[0],covs[0]),multivariate_normal.pdf(x,means[1],covs[1])])
    if max(probs)<AnomalyThreshold:
        result="Anomaly"
        falsePositive += 1
    else:
        result="OK" 
        trueNegative += 1
    #print('Obs: {:2} ({}): Probabilities: [{:.4e},{:.4e}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*probs,result))
printResults(truePositive, trueNegative, falsePositive, falseNegative)



## -- 14 -- ##
from sklearn import svm

print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(i2trainFeaturesNPCA)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(i2trainFeaturesNPCA)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(i2trainFeaturesNPCA)  

tpL, tnL, fpL, fnL = 0, 0, 0, 0
tpRBF, tnRBF, fpRBF, fnRBF = 0, 0, 0, 0
tpP, tnP, fpP, fnP = 0, 0, 0, 0


L1=ocsvm.predict(i3AtestFeaturesNPCA)
L2=rbf_ocsvm.predict(i3AtestFeaturesNPCA)
L3=poly_ocsvm.predict(i3AtestFeaturesNPCA)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=i3AtestFeaturesNPCA.shape
for i in range(nObsTest):
    #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassAttacker[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    
    #Linear
    if AnomResults[L1[i]] == "Anomaly":
        tpL += 1
    else:
        fnL += 1
    #RBF
    if AnomResults[L2[i]] == "Anomaly":
        tpRBF += 1
    else:
        fnRBF += 1
    #Poly
    if AnomResults[L3[i]] == "Anomaly":
        tpP += 1
    else:
        fnP += 1


L1=ocsvm.predict(i3testFeaturesNPCA)
L2=rbf_ocsvm.predict(i3testFeaturesNPCA)
L3=poly_ocsvm.predict(i3testFeaturesNPCA)

nObsTest,nFea=i3testFeaturesNPCA.shape
for i in range(nObsTest):
    #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassClient[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    
    #Linear
    if AnomResults[L1[i]] == "Anomaly":
        fpL += 1
    else:
        tnL += 1
    #RBF
    if AnomResults[L2[i]] == "Anomaly":
        fpRBF += 1
    else:
        tnRBF += 1
    #Poly
    if AnomResults[L3[i]] == "Anomaly":
        fpP += 1
    else:
        tnP += 1
print("\nKernel Linear Statistics")
printResults(tpL, tnL, fpL, fnL)
print("\nKernel RBF Statistics")
printResults(tpRBF, tnRBF, fpRBF, fnRBF)
print("\nKernel Poly Statistics")
printResults(tpP, tnP, fpP, fnP)


## -- 15 -- ##
from sklearn import svm

print('\n-- Anomaly Detection based on One Class Support Vector Machines --')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(i2trainFeaturesN)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(i2trainFeaturesN)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(i2trainFeaturesN)  

tpL, tnL, fpL, fnL = 0,0,0,0
tpRBF, tnRBF, fpRBF, fnRBF = 0,0,0,0
tpP, tnP, fpP, fnP = 0,0,0,0


L1=ocsvm.predict(i3AtestFeaturesN)
L2=rbf_ocsvm.predict(i3AtestFeaturesN)
L3=poly_ocsvm.predict(i3AtestFeaturesN)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=i3AtestFeaturesN.shape
for i in range(nObsTest):
    #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassAttacker[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    
    #Linear
    if AnomResults[L1[i]] == "Anomaly":
        tpL += 1
    else:
        fnL += 1
    #RBF
    if AnomResults[L2[i]] == "Anomaly":
        tpRBF += 1
    else:
        fnRBF += 1
    #Poly
    if AnomResults[L3[i]] == "Anomaly":
        tpP += 1
    else:
        fnP += 1


L1=ocsvm.predict(i3testFeaturesN)
L2=rbf_ocsvm.predict(i3testFeaturesN)
L3=poly_ocsvm.predict(i3testFeaturesN)

nObsTest,nFea=i3testFeaturesN.shape
for i in range(nObsTest):
    #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassClient[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    
    #Linear
    if AnomResults[L1[i]] == "Anomaly":
        fpL += 1
    else:
        tnL += 1
    #RBF
    if AnomResults[L2[i]] == "Anomaly":
        fpRBF += 1
    else:
        tnRBF += 1
    #Poly
    if AnomResults[L3[i]] == "Anomaly":
        fpP += 1
    else:
        tnP += 1
print("\nKernel Linear Statistics")
#printResults(tpL, tnL, fpL, fnL)
print("\nKernel RBF Statistics")
printResults(tpRBF, tnRBF, fpRBF, fnRBF)
print("\nKernel Poly Statistics")
#printResults(tpP, tnP, fpP, fnP)


#-----------------------------------------------------------------------
# A PARTIR DAQUI É CLASSIFICAÇÃO.