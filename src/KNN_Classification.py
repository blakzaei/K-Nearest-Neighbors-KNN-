###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import os

###############################################################################

###############################################################################
#------------------------------------------------------------------------------
#--Handle Missing Data by Mean-------------------------------------------------
def Handle_Missing_Data(df):
    df = df.replace(['?'],np.nan)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(np.round(df.mean()))
    return df
#------------------------------------------------------------------------------

#--Normalize-------------------------------------------------------------------
def Normalize(df):
    
    cols = len(df.columns)
    
    features = df.iloc[:,0:cols-1]
    labels = df.iloc[:,cols-1]
    
    
    features=(features - features.mean())/(features.max() - features.min())
    
    df = features.assign( labels=pd.Series(labels).values)
    
    return df
#------------------------------------------------------------------------------

#--Confusion Matrix------------------------------------------------------------
def Confusion_Matrix(test_set,predictions):
    cm = np.zeros(shape=(2,2))    
    for i in range(len(test_set)):
        if test_set[i][-1] == 1:
            if predictions[i]==1:
                cm[0,0] += 1
            else:
                cm[0,1] += 1
        else: #test_set == 0
            if predictions[i]==1:
                cm[1,0] += 1
            else:
                cm[1,1] += 1
    
    return cm
#------------------------------------------------------------------------------

#-- Accuracy-------------------------------------------------------------------
def Get_Accuracy(test_set, predictions):
    
    correct = 0
    
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1 
            
    acc = (correct/float(len(test_set))) * 100.0
    return acc
#------------------------------------------------------------------------------

#--Draw Confusion Matrix-------------------------------------------------------
def Draw_Confusion_Matrix(cm,k):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.imshow(cm, origin='lower', interpolation='None', cmap='viridis' , alpha=0.5)
    ax.text(0, 0, s=cm[1][0], color='black', ha='center', va='center')
    ax.text(0, 1, s=cm[0][0], color='black', ha='center', va='center')
    ax.text(1, 0, s=cm[1][1], color='black', ha='center', va='center')
    ax.text(1, 1, s=cm[0][1], color='black', ha='center', va='center')
    plt.rcParams.update({'font.size': 22})
    plt.axis('off')
    plt.title('K= ' + str(k))
    plt.show()
#------------------------------------------------------------------------------


#-- Run KNN with K Fold Cross Validation --------------------------------------
def KNN_With_K_Fold(df,number_of_folds,k_neighbors, distance_type= 'euclidean'):
    
    df_length = len(df)    
    fold_size = df_length//number_of_folds   
    
    acc_values = []
    final_cm = np.zeros(shape=(2,2))  
    
    for fold in range(number_of_folds):
        
        #-- set indexes for split data --
        start_index = fold * fold_size
        end_index = ((fold+1) * fold_size) -1
        if (df_length-1)-end_index < fold_size:
            end_index = df_length -1   
        
        #-- split test data ( 1 fold ) --
        test_set = df.iloc[start_index:end_index+1,:].values   
        
        #-- split train data ( number_of_folds -1 ) --
        l_drop = end_index - start_index +1
        train_set = df
        for i in range(l_drop):
            train_set = train_set.drop([train_set.index[start_index]])
        
        train_set = train_set.iloc[:,:].values    
        
        #-- run KNN --      
        acc , cm = KNN(train_set, test_set , k_neighbors , distance_type )
        
        acc_values.append(acc)
        final_cm += cm        
        
    
    acc_values = np.array(acc_values)
    return acc_values.mean() , final_cm


#-- KNN------------------------------------------------------------------------
def KNN(train_set, test_set, k, distance_type='euclidean'):
    
    predictions=[]

    for i in range(len(test_set)):
        test_instance = test_set[i]       
        predicted_label = Predict(train_set, test_instance, k, distance_type)        
        predictions.append(predicted_label) 
        
    acc = Get_Accuracy(test_set, predictions)   
    cm = Confusion_Matrix(test_set, predictions)
    
    return acc , cm
#------------------------------------------------------------------------------

#-- Predict Label by voting ---------------------------------------------------
def Predict(train_set,test_instance,k, distance_type='euclidean'):
    
    #-- Find k nearest neighbors --
    neighbors = Get_K_Nearest_Neighbors(train_set, test_instance, k, distance_type)
    
    #-- labels of nearest neighbors --
    labels = [row[-1] for row in neighbors]
    
    #-- set target label by voting --
    prediction = max(set(labels), key=labels.count)
    
    return prediction
#------------------------------------------------------------------------------       
        
#-- Get K Nearest Neighbors----------------------------------------------------
def Get_K_Nearest_Neighbors(train_set, test_instance, k, distance_type='euclidean'):
    
    distances = []
    
    #-- distance between test instance and all train instances --
    for i in range(len(train_set)):
        if distance_type == 'euclidean':
            dist = Euclidean_Distance(test_instance, train_set[i])
            
        elif distance_type == 'manhattan':
            dist = Manhattan_Distance(test_instance, train_set[i])
        
        elif distance_type == 'cosine':
            dist = Cosine_Distance(test_instance, train_set[i])
            
            
        distances.append((train_set[i], dist))    
    
    #-- sort neighbors by distance --
    distances.sort(key=lambda x: x[1])   
    
    #-- get index of k nearest neighbors --
    k_neighbors = []
    for i in range(k):
        k_neighbors.append(distances[i][0])
        
    return k_neighbors
#------------------------------------------------------------------------------   
        
#--Euclidean Distance----------------------------------------------------------
def Euclidean_Distance(X1, X2):
    distance = 0.0   
    for i in range(len(X1)-1):
        distance += (X1[i] - X2[i])**2
        
    return sqrt(distance)
#------------------------------------------------------------------------------

#--Manhattan Distance----------------------------------------------------------
def Manhattan_Distance(X1, X2):
    distance = 0.0
    for i in range(len(X1)-1):
        distance += abs(X1[i] - X2[i])
    return distance
#------------------------------------------------------------------------------

#--Cosine Distance----------------------------------------------------------
def Cosine_Distance(X1, X2):
    m = 0
    for i in range(len(X1)-1):
        m += X1[i] * X2[i]
        
    l1 = 0
    for i in range(len(X1)-1):
        l1 += X1[i] **2
    
    l2 = 0
    for i in range(len(X2)-1):
        l2 += X2[i] **2
    
    distance = m / (l1*l2)
    return distance
#------------------------------------------------------------------------------


###############################################################################

###############################################################################

#-- Load DS -------------------------------------------------------------------
path = os.path.dirname(os.getcwd()) + '\dataset\mammographic_masses\mammographic_masses.data'

attributes_name = ['BI-RADS','Age','Shape','Margin','Density','Severity']

df = pd.read_csv(path,
                 sep=',',
                 header=None,
                 names=attributes_name)

#-- Replace Missing Data with Mean --------------------------------------------
df = Handle_Missing_Data(df)

#-- Normalize Features -------------------------------------------------------
df = Normalize(df)


#--Shuffle --------------------------------------------------------------------
df = df.sample(frac=1)

###############################################################################

###############################################################################

#-- Test different k_values ---------------------------------------------------
k_values = [1,3,5,7,15,30]
acc_values = []

#-- log --
print("Testing Different K values--------------------------------")

for i in range(len(k_values)):
    
    #-- log --
    print("\n\tk=%d ---" %k_values[i])
    
    acc , final_cm = KNN_With_K_Fold(df, number_of_folds=10, k_neighbors=k_values[i])
    acc_values.append(acc)
    Draw_Confusion_Matrix(final_cm,k_values[i])

plt.scatter(x=k_values , y= acc_values)
plt.plot(k_values,acc_values)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.show()


# Test different Distance Measures --------------------------------------------
k = 15
acc_values = []
distance_measures = ['euclidean','manhattan','cosine']

#-- log --
print("Testing Different Distance Measures --------------------------------")

for i in range(len(distance_measures)):
    
    #-- log --
    print("\n--- %s ---" %distance_measures[i])
    
    acc , cm = KNN_With_K_Fold(df, number_of_folds=10,
                                       k_neighbors=k,
                                       distance_type= distance_measures[i])
    
    Draw_Confusion_Matrix(cm, k)
    acc_values.append(acc)

plt.scatter(x=distance_measures , y= acc_values)
plt.plot(distance_measures, acc_values)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.xticks(distance_measures)
plt.show()


