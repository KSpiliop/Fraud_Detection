
# Tuning XGBoost hyper-parameters with Simulated Annealing  
# An example in Credit Card Fraud Detection


The purpose of this experiment is to show how heuristics such as [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing "Simulated Annealing") can be used to find good combinations of many hyper-parameters efficiently. This approach is obviously better than blind random generation. It is also preferable to fine-tuning each hyper-parameter separately because there are interactions between the hyper-parameters. 

The XGBoost algorithm is a good case because it has many hyper-parameters. Exhaustive grid search can be computationally prohibitive.   

For a very good discussion of the theoretical details of XGBoost, see this [Slideshare presentation](https://www.slideshare.net/ShangxuanZhang/kaggle-winning-solution-xgboost-algorithm-let-us-learn-from-its-author "Slideshare presentation") of the algorithm with title "*Kaggle Winning Solution Xgboost algorithm -- Let us learn from its author*" by Tianqi Chen.


## Loading the data

This is a Kaggle dataset taken from [here](https://www.kaggle.com/dalpozz/creditcardfraud "Kaggle dataset") which contains credit card transactions data and a fraud flag. It appeared originally in [Dal Pozzolo, Andrea, et al. "Calibrating Probability with Undersampling for Unbalanced Classification." Computational Intelligence, 2015 IEEE Symposium Series on. IEEE, 2015](http://www.oliviercaelen.be/doc/SSCI_calib_final.pdf "Original paper"). There is a Time variable (seconds from the first transaction in the dataset), an Amount variable, the Class variable (1=fraud, 0= no fraud) and the rest (V1-V28) are factor variables obtained through Principal Components Analysis from the original variables.

This is not a very difficult case for XGBoost as it will be seen. The main objective in this experiment is to show that the heuristic search finds a suitable set of hyper-parameters out of a quite large set of combinations.

We can verify below that this is a highly imbalanced dataset, typical of fraud detection data. We will take this into account when setting the weights of observations in XGBoost parameters. 

Since we have plenty of data we are going to calibrate the hyper-parameters on a validation dataset and evaluate performance on an unseen testing dataset. 


```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('ggplot')

dat = pd.read_csv('creditcard.csv')

print(dat.head())
print('\nThe distribution of the target variable:\n')
dat['Class'].value_counts()

```

       Time        V1        V2        V3        V4        V5        V6        V7  \
    0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
    1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
    2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
    3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
    4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   
    
             V8        V9  ...         V21       V22       V23       V24  \
    0  0.098698  0.363787  ...   -0.018307  0.277838 -0.110474  0.066928   
    1  0.085102 -0.255425  ...   -0.225775 -0.638672  0.101288 -0.339846   
    2  0.247676 -1.514654  ...    0.247998  0.771679  0.909412 -0.689281   
    3  0.377436 -1.387024  ...   -0.108300  0.005274 -0.190321 -1.175575   
    4 -0.270533  0.817739  ...   -0.009431  0.798278 -0.137458  0.141267   
    
            V25       V26       V27       V28  Amount  Class  
    0  0.128539 -0.189115  0.133558 -0.021053  149.62      0  
    1  0.167170  0.125895 -0.008983  0.014724    2.69      0  
    2 -0.327642 -0.139097 -0.055353 -0.059752  378.66      0  
    3  0.647376 -0.221929  0.062723  0.061458  123.50      0  
    4 -0.206010  0.502292  0.219422  0.215153   69.99      0  
    
    [5 rows x 31 columns]
    
    The distribution of the target variable:
    
    




    0    284315
    1       492
    Name: Class, dtype: int64



## Data exploration.

Although the context of most of the variables is not known (recall, V1-V28 are factors summarizing the transactional data), we know that V1-V28 are by construction standardized, with a mean of 0 and and a standard deviation of 1. We standardize the Time and Amount variables too.

The data exploration and in particular Welch’s t-tests reveal that almost all the factors are significantly associated with the Class variable. The mean of these variables is almost zero in Class 0 and clearly non-zero in Class 1. The Time and Amount variables are also significant. There does not seem to by any reason for variable selection.

Some of the factors (and the Amount variable) are quite skewed and have very thin distributions. If we were to apply some other method (say, logistic regression) we could apply some transformations (and probably binning) but XGBoost is insensitive to such deviations from normality.  


```python

from scipy import stats
from sklearn import preprocessing

dat['Time'] = preprocessing.scale(dat['Time'])
dat['Amount'] = preprocessing.scale(dat['Amount'])

print('\nMeans of variables in the two Class categories:\n')
pt = pd.pivot_table(dat, values=dat.columns[:30], columns = 'Class', aggfunc='mean')
print(pt.loc[dat.columns])

print('\nP-values of Welch’s t-tests and shape statistics:\n')
for i in range(30):
    col_name = dat.columns[i]
    t, p_val = stats.ttest_ind(dat.loc[ dat['Class']==0, col_name], dat.loc[ dat['Class']==1, col_name],equal_var=False)  
    skewness = dat.loc[:,col_name].skew()
    kurtosis = stats.kurtosis(dat.loc[:,col_name])
    print('Variable: {:7s}'.format(col_name),end='')    
    print('p-value: {:6.3f}  skewness: {:6.3f}  kurtosis: {:6.3f}'.format(p_val, skewness, kurtosis))

    
fig, axes = plt.subplots(nrows=6, ncols=5,figsize=(10,10))
axes = axes.flatten()

columns = dat.columns
for i in range(30):
  axes[i].hist(dat[columns[i]], bins=50,facecolor='b',alpha=0.5)
  axes[i].set_title(columns[i])
  axes[i].set_xlim([-4., +4.])
  plt.setp(axes[i].get_xticklabels(), visible=False) 
  plt.setp(axes[i].get_yticklabels(), visible=False) 


```

    
    Means of variables in the two Class categories:
    
    Class          0         1
    Time    0.000513 -0.296223
    V1      0.008258 -4.771948
    V2     -0.006271  3.623778
    V3      0.012171 -7.033281
    V4     -0.007860  4.542029
    V5      0.005453 -3.151225
    V6      0.002419 -1.397737
    V7      0.009637 -5.568731
    V8     -0.000987  0.570636
    V9      0.004467 -2.581123
    V10     0.009824 -5.676883
    V11    -0.006576  3.800173
    V12     0.010832 -6.259393
    V13     0.000189 -0.109334
    V14     0.012064 -6.971723
    V15     0.000161 -0.092929
    V16     0.007164 -4.139946
    V17     0.011535 -6.665836
    V18     0.003887 -2.246308
    V19    -0.001178  0.680659
    V20    -0.000644  0.372319
    V21    -0.001235  0.713588
    V22    -0.000024  0.014049
    V23     0.000070 -0.040308
    V24     0.000182 -0.105130
    V25    -0.000072  0.041449
    V26    -0.000089  0.051648
    V27    -0.000295  0.170575
    V28    -0.000131  0.075667
    Amount -0.000234  0.135382
    Class        NaN       NaN
    
    P-values of Welch’s t-tests and shape statistics:
    
    Variable: Time   p-value:  0.000  skewness: -0.036  kurtosis: -1.294
    Variable: V1     p-value:  0.000  skewness: -3.281  kurtosis: 32.486
    Variable: V2     p-value:  0.000  skewness: -4.625  kurtosis: 95.771
    Variable: V3     p-value:  0.000  skewness: -2.240  kurtosis: 26.619
    Variable: V4     p-value:  0.000  skewness:  0.676  kurtosis:  2.635
    Variable: V5     p-value:  0.000  skewness: -2.426  kurtosis: 206.901
    Variable: V6     p-value:  0.000  skewness:  1.827  kurtosis: 42.642
    Variable: V7     p-value:  0.000  skewness:  2.554  kurtosis: 405.600
    Variable: V8     p-value:  0.063  skewness: -8.522  kurtosis: 220.583
    Variable: V9     p-value:  0.000  skewness:  0.555  kurtosis:  3.731
    Variable: V10    p-value:  0.000  skewness:  1.187  kurtosis: 31.988
    Variable: V11    p-value:  0.000  skewness:  0.357  kurtosis:  1.634
    Variable: V12    p-value:  0.000  skewness: -2.278  kurtosis: 20.241
    Variable: V13    p-value:  0.028  skewness:  0.065  kurtosis:  0.195
    Variable: V14    p-value:  0.000  skewness: -1.995  kurtosis: 23.879
    Variable: V15    p-value:  0.050  skewness: -0.308  kurtosis:  0.285
    Variable: V16    p-value:  0.000  skewness: -1.101  kurtosis: 10.419
    Variable: V17    p-value:  0.000  skewness: -3.845  kurtosis: 94.798
    Variable: V18    p-value:  0.000  skewness: -0.260  kurtosis:  2.578
    Variable: V19    p-value:  0.000  skewness:  0.109  kurtosis:  1.725
    Variable: V20    p-value:  0.000  skewness: -2.037  kurtosis: 271.011
    Variable: V21    p-value:  0.000  skewness:  3.593  kurtosis: 207.283
    Variable: V22    p-value:  0.835  skewness: -0.213  kurtosis:  2.833
    Variable: V23    p-value:  0.571  skewness: -5.875  kurtosis: 440.081
    Variable: V24    p-value:  0.000  skewness: -0.552  kurtosis:  0.619
    Variable: V25    p-value:  0.249  skewness: -0.416  kurtosis:  4.290
    Variable: V26    p-value:  0.015  skewness:  0.577  kurtosis:  0.919
    Variable: V27    p-value:  0.006  skewness: -1.170  kurtosis: 244.985
    Variable: V28    p-value:  0.002  skewness: 11.192  kurtosis: 933.381
    Variable: Amount p-value:  0.004  skewness: 16.978  kurtosis: 845.078
    


![png](output_3_1.png)


## Data partitioning.

In this step, we partition the dataset into 40% training, 30% validation and 30% testing. Note the use of the random.shuffle() function from numpy. We also make the corresponding matrices **train**, **valid** and **test** containing predictors only with labels **trainY**, **validY** and **testY**, respectively.


```python
import random

random.seed(1234)

Class = dat['Class'].values
dat2 = dat.drop(['Class'], axis=1)

allIndices = np.arange(len(Class))
np.random.shuffle(allIndices)

numTrain = int(round(0.40*len(Class)))
numValid = int(round(0.30*len(Class)))
numTest = len(Class)-numTrain-numValid

inTrain = sorted(allIndices[:numTrain])
inValid = sorted(allIndices[numTrain:(numTrain+numValid)])
inTest =  sorted(allIndices[(numTrain+numValid):])

train = dat2.iloc[inTrain,:]
valid= dat2.iloc[inValid,:]
test =  dat2.iloc[inTest,:]

trainY = Class[inTrain]
validY = Class[inValid]
testY = Class[inTest]
```

## Preparing the Booster: Fixed parameters.

First we create the matrices in the format required by XGBoost with the xgb.DMatrix() function, passing for each dataset the predictors data and the labels. Then we set some fixed parameters. The number of boosting iterations (num_rounds) is set to 10. We initialize the **param dictionary** with silent=1 (no messages). Parameter min_child_weight is set at the default value of 1. This is the minimum weighted number of observations in a child node for further partitioning. The objective is binary classification and the evaluation metric is the Area Under Curve (AUC), the default for binary classification. In a more advanced implementation we would make a customized evaluation function, as described in [XGBoost API](http://xgboost.readthedocs.io/en/latest/python/python_api.html "XGBoost API").

We are going to **expand the param dictionary** with the parameters in the Simulated Annealing search.


```python
import xgboost as xgb

dtrain = xgb.DMatrix(train, label=trainY)
dvalid = xgb.DMatrix(valid, label=validY)
dtest = xgb.DMatrix(test, label=testY)

## fixed parameters
num_rounds=10 # number of boosting iterations

param = {'silent':1,
         'min_child_weight':1,
         'objective':'binary:logistic',
         'eval_metric':'auc'}  
```

## Preparing the Booster: Variable parameters 

In what follows we combine the suggestions from several sources, notably:

1. [The official XGBoost documentation](http://xgboost.readthedocs.io/en/latest/parameter.html "Official XGBoost Guide") and in particular the [Notes on Parameter Tuning](http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html "Notes on Parameter Tuning")

2. [The article "Complete Guide to Parameter Tuning in XGBoost" from Analytics Vidhya](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ "Analytics Vidhya")  

3. [Another Slideshare presentation with title "Open Source Tools & Data Science Competitions"](https://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1 "Second Slideshare presentation")

We select several important parameters for the heuristic search:

* **max_depth**: the maximum depth of a tree, in [1,∞], with default value 6. This is highly data-dependent. [2] quotes as typical values: 3-10 and [3] advises to start from 6. We choose to explore also larger values and select 5-30 levels in steps of 5.
* **subsample**, in (0,1] with default value 1. This is the proportion of the training instances used in trees and smaller values can prevent over-fitting. In [2] values in 0.5-1 are suggested. [3] suggests to leave this at 1. We decide to test values in 0.5-1.0 in steps of 0.1.
* **colsample_bytree**, in (0,1] with default value 1. This is the subsample ratio of columns (features) used to construct a tree. In [2] values in 0.5-1 are suggested. The advice in [3] is 0.3-0.5. We will try similar values as with subsample.
* **eta** (or **learning_rate**), in [0,1], with default value 0.3. This is the shrinking rate of the feature weights and larger values (but not too high!) can be used to prevent overfitting. A suggestion in [2] is to use values in 0.01-0.2. We are generous and select the whole range in [0.1,0.4] in steps of 0.05.
* **gamma**, in [0, ∞], with default value 0. This is the minimum loss function reduction required for a split. [3] suggests to leave this at 0. We can experiment with values in 0-2 by steps of 0.05.
* **scale_pos_weight** which controls the balance of positive and negative weights with default value 1. The advice in [1] is to use the ratio of negative to positive cases which is 503 here, i.e. to put a weight that large to the positive cases. [2] similarly suggests a large value in case of high class imbalance as is the case here. We can try some small values and some larger ones. 


The **total number of possible combinations** is 52920 and we are only going to test a small fraction of 100 of them, as many as the number of the Simulated Annealing iterations. 

We also initialize a dataframe which will hold the results, for examination and also to avoid repetitions of combinations during the heuristic search.



```python

from collections import OrderedDict

scale_pos_weight_suggestion = sum(trainY==0)/sum(trainY==1)  ## = 608
print('Ratio of negative to positive instances: {:6.1f}'.format(scale_pos_weight_suggestion))
event_rate = sum(trainY==1)/len(train)
min_child_weight_suggestion = 1/np.sqrt(event_rate)
print('1/sqrt(event rate): {:6.1f}'.format(min_child_weight_suggestion))

## parameters to be tuned
tune_dic = OrderedDict()

tune_dic['max_depth']= [5,10,15,20,25,30] ## maximum tree depth
tune_dic['subsample']=[0.5,0.6,0.7,0.8,0.9,1.0]
tune_dic['colsample_bytree']= [0.5,0.6,0.7,0.8,0.9,1.0] ## subsample ratio of columns
tune_dic['eta']= [0.10,0.15,0.20,0.25,0.30,0.35,0.40]  ## learning rate
tune_dic['gamma']= [0.00,0.05,0.10,0.15,0.20]  ##
tune_dic['scale_pos_weight']=[30,40,50,300,400,500,600,700] 

lengths = [len(lst) for lst in tune_dic.values()]

combs=1
for i in range(len(lengths)):
    combs *= lengths[i]
print('Total number of combinations: {:16d}'.format(combs))  

maxiter=100
columns=[*tune_dic.keys()]+['F-Score','Best F-Score','Duplicate']
results = pd.DataFrame(index=range(maxiter), columns=columns) ## to check results

```

    Ratio of negative to positive instances:  560.2
    1/sqrt(event rate):   23.7
    Total number of combinations:            60480
    

## Functions for training and performance reporting.

Next we define two functions:

<blockquote>Function **perf_measures()** accepts some predictions and labels, optionally prints the confusion matrix, and returns the [F-Score](https://en.wikipedia.org/wiki/F1_score "F-Score") This is a measure of performance combining [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall "Precision and recall") and will guide the heuristic search.</blockquote> 

<blockquote>Function **do_train()** accepts as parameters:  
&nbsp;  
- the current choice of variable parameters in a dictionary (cur_choice),   
- the full dictionary of parameters to be passed to the main XGBoost training routine (param),   
- a train dataset in XGBoost format (train),   
- a string identifier (train_s),   
- its labels (trainY),  
- the corresponding arguments for a validation dataset (valid, valid_s, validY),  
- and the option to print the confusion matrix (print_conf_matrix).   
&nbsp;  
It then trains the model and returns the F-score of the predictions on the validation dataset together with the model. The call to the main train routine **xgb.train()** has the following arguments:   
&nbsp;    
- the full dictionary of the parameters (param),    
- the train dataset in XGBoost format (train),  
- the number of boosting iterations  (num_boost),  
- a watchlist with datasets information to show progress (evals), 
- the frequency of reporting (verbose_eval), here set to a high value to report only in the first and last iteration.
</blockquote> 


```python
def perf_measures(preds, labels, print_conf_matrix=False):
    
    act_pos=sum(labels==1) ## actual positive
    act_neg=len(labels) - act_pos ## actual negative
    
    pred_pos=sum(1 for i in range(len(preds)) if (preds[i]>=0.5)) ## predicted positive
    true_pos=sum(1 for i in range(len(preds)) if (preds[i]>=0.5) & (labels[i]==1)) ## predicted negative
    
    false_pos=pred_pos - true_pos ## false positive
    false_neg=act_pos-true_pos ## false negative
    true_neg=act_neg-false_pos ## true negative
      
    precision = true_pos/pred_pos ## tp/(tp+fp) percentage of correctly classified predicted positives
    recall = true_pos/act_pos ## tp/(tp+fn) percentage of positives correctly classified
    
    f_score = 2*precision*recall/(precision+recall) 
    
    if print_conf_matrix:
        print('\nconfusion matrix')
        print('----------------')
        print( 'tn:{:6d} fp:{:6d}'.format(true_neg,false_pos))
        print( 'fn:{:6d} tp:{:6d}'.format(false_neg,true_pos))
    
    return(f_score)


def do_train(cur_choice, param, train,train_s,trainY,valid,valid_s,validY,print_conf_matrix=False):
    ## train with given fixed and variable parameters
    ## and report the F-score on the validation dataset
    
    print('Parameters:')
    for (key,value) in cur_choice.items():
        print(key,': ',value,' ',end='')
        param[key]=value
    print('\n')    
    
    evallist  = [(train,train_s), (valid,valid_s)]
    model = xgb.train( param, train, num_boost_round=num_rounds,
                      evals=evallist,verbose_eval=50)    
    preds = model.predict(valid)
    labels = valid.get_label()
      
    f_score = perf_measures(preds, labels,print_conf_matrix)
    
    return(f_score, model)    

```

## Producing neighbouring combinations.

Next we define a function next_choice() which either produces a random combination of the variable hyper-parameters (if no current parameters are passed) or generates a neighbour combination of hyper-parameters passed in cur_params.   

In the second case we first select at random a parameter to be changed. Then:
* If this parameter currently has the smallest value, we select the next one.
* If this parameter currently has the largest value, we select the previous one.
* Otherwise, we select the left (smaller) or right (larger) value randomly.

Repetitions are avoided in the function which carries out the Simulated Annealing search.


```python


def next_choice(cur_params=None):
    ## returns a random combination of the variable hyper-parameters (if cur_params=None)
    ## or a random neighboring combination from cur_params
    if cur_params:
        ## chose parameter to change
        choose_param_index = np.random.randint(len(cur_params)) ## index of parameter
        choose_param_name =[*tune_dic.keys()][choose_param_index] ## parameter name 
        
        cur_value = cur_params[choose_param_name]
        all_values =  list(tune_dic[choose_param_name])
        cur_index = all_values.index(cur_value)
        if cur_index==0: ## if it is the first in the range select the second one
            next_index=1
        elif cur_index==len(all_values)-1: ## if it is the last in the range select the previous one
            next_index=len(all_values)-2
        else: ## otherwise select the left or right value randomly
            direction=np.random.randint(2)
            if direction==0:
                next_index=cur_index-1
            else:
                next_index=cur_index+1
        next_params = cur_params
        next_params[choose_param_name] = all_values[next_index]
        print('selected in next_choice: {:10s}: from {:6.2f} to {:6.2f}'.
              format([*tune_dic.keys()][choose_param_index],all_values[cur_index],all_values[next_index] ))
    else:
        next_params=dict()
        for i in range(len(tune_dic)):
            key = [*tune_dic.keys()][i] 
            values = [*tune_dic.values()][i]
            next_params[key] = np.random.choice(values)
    return(next_params)  

```

## Application of the Simulated Annealing algorithm.

At each iteration of the Simulated Annealing algorith, one combination of hyper-parameters is selected. The XGBoost algorithm is trained with these parameters and the F-score on the validation set is produced. 

* If this F-score is better (larger) than the one at the previous iteration, i.e. there is a "local" improvement, the combination is accepted as the current combination and a neighbouring combination is selected for the next iteration.
* Otherwise, i.e. if this F-score is worse (smaller) than the one at the previous iteration and the decline is Δf < 0, the combination is accepted as the current one with probability exp(-beta Δf/T) where beta is a constant (here beta = 1.2) and T is the current "temperature". The idea is that we start with a high temperature and "bad" solutions are easily accepted at first, in the hope of exploring wide areas of the search space. But as the temperature drops, bad solutions are less likely to be accepted and the search becomes more focused.      

The temperature starts at a fixed value T0 and is reduced by a factor of alpha < 1 every n number of iterations. Here T0 = 0.2, n=5 and a = 0.85.   

The selection of the parameters of this "cooling schedule" can be done easily in MS Excel. In this example we select the average acceptance probabilities for F-Score deterioration of 0.15, 0.10, 0.05, 0.02, 0.01 during the first, second,...,fifth 20 iterations respectively. We set all to 30% and use Solver to find suitable parameters. 

A **warning**: if the number of iterations is not suficiently smaller than the total number of combinations, there may be too many  repetitions and delays. The simple approach for producing combinations here does not address such cases.  



```python
import time

t0 = time.clock()

T=0.2
best_params = dict() ## initialize dictionary to hold the best parameters

best_f_score = -1. ## initialize best f-score
prev_f_score = -1. ## initialize previous f-score
prev_choice = None ## initialize previous selection of parameters

for iter in range(maxiter):
    print('\nIteration = {:5d}  T = {:12.6f}'.format(iter,T))

    ## find next selection of parameters not visited before
    ## this is not im
    while True:
        cur_choice=next_choice(prev_choice) ## first selection or selection - neighbour of prev_choice
        
        ## check if selection has already been visited
        tmp=abs(results.loc[:,[*cur_choice.keys()]] - list(cur_choice.values()))
        tmp=tmp.sum(axis=1)
        #tmp=results.loc[:,[*cur_choice.keys()]].as_matrix()==cur_choice.values()
        #tmp=tmp.sum(axis=1)
        if any(tmp==0): ## selection has already been visited
            print('\nCombination revisited - searching again')
        else:
            break ## break out the while-loop
    
    
    ## train the model and obtain f-score on the validation dataset
    f_score,model=do_train(cur_choice, param, dtrain,'train',trainY,dvalid,'valid',validY)
    
    ## store parameters and results
    results.loc[iter,[*cur_choice.keys()]]=list(cur_choice.values())
    
    #if any(results.loc[:iter,].duplicated([*tune_dic.keys()],keep=False)):
    #    raise Exception('Duplicate!')
    
    print('\nF-Score: {:6.2f}  previous: {:6.2f}  best so far: {:6.2f}'.format(f_score, prev_f_score, best_f_score))
 
    if f_score > prev_f_score:
        print('Local improvement')
        
        ## accept this combination as the new starting point
        prev_f_score = f_score
        prev_choice = cur_choice
        
        ## update best parameters if the f-score is globally better
        if f_score > best_f_score:
            
            best_f_score = f_score
            print('*** Best f-score updated')
            for (key,value) in prev_choice.items():
                print(key,': ',value,' ',end='')
                best_params[key]=value

    else: ## f-score is smaller than the previous one
        
        print('Worse result')
        ## this combination as the new starting point with probability exp(-(1.2 x f-score decline)/temperature) 
        rnd = random.random()
        diff = f_score-prev_f_score
        thres=np.exp(1.2*diff/T)
        if rnd <= thres:
            print('\nF-Score decline: {:8.4f}  threshold: {:6.4f}  random number: {:6.4f} -> accepted'.
                  format(diff, thres, rnd))
            prev_f_score = f_score
            prev_choice = cur_choice
 
        else:
            print('\nF-Score decline: {:8.4f}  threshold: {:6.4f}  random number: {:6.4f} -> rejected'.
                 format(diff, thres, rnd))
    results.loc[iter,'F-Score']=f_score
    results.loc[iter,'Best F-Score']=best_f_score
    if iter % 5 == 0: T=0.85*T   
        
print('\n',time.clock() - t0, ' seconds process time\n')    

print('Best variable parameters found:\n')
print(best_params)

```

    
    Iteration =     0  T =     0.200000
    Parameters:
    eta :  0.15  subsample :  0.9  gamma :  0.2  scale_pos_weight :  50  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.982071	valid-auc:0.934483
    [9]	train-auc:0.99991	valid-auc:0.974861
    
    F-Score:   0.77  previous:  -1.00  best so far:  -1.00
    Local improvement
    *** Best f-score updated
    eta :  0.15  subsample :  0.9  gamma :  0.2  scale_pos_weight :  50  max_depth :  10  colsample_bytree :  0.8  
    Iteration =     1  T =     0.170000
    selected in next_choice: scale_pos_weight: from  50.00 to  40.00
    Parameters:
    eta :  0.15  subsample :  0.9  gamma :  0.2  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.981733	valid-auc:0.940463
    [9]	train-auc:0.999906	valid-auc:0.974748
    
    F-Score:   0.78  previous:   0.77  best so far:   0.77
    Local improvement
    *** Best f-score updated
    eta :  0.15  subsample :  0.9  gamma :  0.2  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.8  
    Iteration =     2  T =     0.170000
    selected in next_choice: scale_pos_weight: from  40.00 to  50.00
    
    Combination revisited - searching again
    selected in next_choice: gamma     : from   0.20 to   0.15
    Parameters:
    eta :  0.15  subsample :  0.9  gamma :  0.15  scale_pos_weight :  50  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.982071	valid-auc:0.934483
    [9]	train-auc:0.99991	valid-auc:0.974861
    
    F-Score:   0.77  previous:   0.78  best so far:   0.78
    Worse result
    
    F-Score decline:  -0.0064  threshold: 0.9556  random number: 0.9665 -> rejected
    
    Iteration =     3  T =     0.170000
    selected in next_choice: scale_pos_weight: from  50.00 to 300.00
    Parameters:
    eta :  0.15  subsample :  0.9  gamma :  0.15  scale_pos_weight :  300  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.993044	valid-auc:0.976564
    [9]	train-auc:0.999913	valid-auc:0.949631
    
    F-Score:   0.74  previous:   0.78  best so far:   0.78
    Worse result
    
    F-Score decline:  -0.0366  threshold: 0.7721  random number: 0.4407 -> accepted
    
    Iteration =     4  T =     0.170000
    selected in next_choice: colsample_bytree: from   0.80 to   0.90
    Parameters:
    eta :  0.15  subsample :  0.9  gamma :  0.15  scale_pos_weight :  300  max_depth :  10  colsample_bytree :  0.9  
    
    [0]	train-auc:0.991818	valid-auc:0.963271
    [9]	train-auc:0.999907	valid-auc:0.929012
    
    F-Score:   0.72  previous:   0.74  best so far:   0.78
    Worse result
    
    F-Score decline:  -0.0214  threshold: 0.8600  random number: 0.0075 -> accepted
    
    Iteration =     5  T =     0.170000
    selected in next_choice: colsample_bytree: from   0.90 to   0.80
    
    Combination revisited - searching again
    selected in next_choice: eta       : from   0.15 to   0.20
    Parameters:
    eta :  0.2  subsample :  0.9  gamma :  0.15  scale_pos_weight :  300  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.993044	valid-auc:0.976564
    [9]	train-auc:0.999912	valid-auc:0.946879
    
    F-Score:   0.75  previous:   0.72  best so far:   0.78
    Local improvement
    
    Iteration =     6  T =     0.144500
    selected in next_choice: eta       : from   0.20 to   0.25
    Parameters:
    eta :  0.25  subsample :  0.9  gamma :  0.15  scale_pos_weight :  300  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.993044	valid-auc:0.976564
    [9]	train-auc:0.999914	valid-auc:0.960358
    
    F-Score:   0.77  previous:   0.75  best so far:   0.78
    Local improvement
    
    Iteration =     7  T =     0.144500
    selected in next_choice: eta       : from   0.25 to   0.20
    
    Combination revisited - searching again
    selected in next_choice: scale_pos_weight: from 300.00 to  50.00
    Parameters:
    eta :  0.2  subsample :  0.9  gamma :  0.15  scale_pos_weight :  50  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.982071	valid-auc:0.934483
    [9]	train-auc:0.99991	valid-auc:0.973333
    
    F-Score:   0.78  previous:   0.77  best so far:   0.78
    Local improvement
    *** Best f-score updated
    eta :  0.2  subsample :  0.9  gamma :  0.15  scale_pos_weight :  50  max_depth :  10  colsample_bytree :  0.8  
    Iteration =     8  T =     0.144500
    selected in next_choice: max_depth : from  10.00 to   5.00
    Parameters:
    eta :  0.2  subsample :  0.9  gamma :  0.15  scale_pos_weight :  50  max_depth :  5  colsample_bytree :  0.8  
    
    [0]	train-auc:0.946664	valid-auc:0.923623
    [9]	train-auc:0.989112	valid-auc:0.983706
    
    F-Score:   0.77  previous:   0.78  best so far:   0.78
    Worse result
    
    F-Score decline:  -0.0173  threshold: 0.8661  random number: 0.9110 -> rejected
    
    Iteration =     9  T =     0.144500
    selected in next_choice: subsample : from   0.90 to   0.80
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.15  scale_pos_weight :  50  max_depth :  5  colsample_bytree :  0.8  
    
    [0]	train-auc:0.957383	valid-auc:0.94888
    [9]	train-auc:0.992577	valid-auc:0.98219
    
    F-Score:   0.81  previous:   0.78  best so far:   0.78
    Local improvement
    *** Best f-score updated
    eta :  0.2  subsample :  0.8  gamma :  0.15  scale_pos_weight :  50  max_depth :  5  colsample_bytree :  0.8  
    Iteration =    10  T =     0.144500
    selected in next_choice: colsample_bytree: from   0.80 to   0.70
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.15  scale_pos_weight :  50  max_depth :  5  colsample_bytree :  0.7  
    
    [0]	train-auc:0.957381	valid-auc:0.948879
    [9]	train-auc:0.989316	valid-auc:0.981458
    
    F-Score:   0.82  previous:   0.81  best so far:   0.81
    Local improvement
    *** Best f-score updated
    eta :  0.2  subsample :  0.8  gamma :  0.15  scale_pos_weight :  50  max_depth :  5  colsample_bytree :  0.7  
    Iteration =    11  T =     0.122825
    selected in next_choice: colsample_bytree: from   0.70 to   0.60
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.15  scale_pos_weight :  50  max_depth :  5  colsample_bytree :  0.6  
    
    [0]	train-auc:0.957356	valid-auc:0.948817
    [9]	train-auc:0.989536	valid-auc:0.978118
    
    F-Score:   0.83  previous:   0.82  best so far:   0.82
    Local improvement
    *** Best f-score updated
    eta :  0.2  subsample :  0.8  gamma :  0.15  scale_pos_weight :  50  max_depth :  5  colsample_bytree :  0.6  
    Iteration =    12  T =     0.122825
    selected in next_choice: subsample : from   0.80 to   0.90
    Parameters:
    eta :  0.2  subsample :  0.9  gamma :  0.15  scale_pos_weight :  50  max_depth :  5  colsample_bytree :  0.6  
    
    [0]	train-auc:0.946661	valid-auc:0.923778
    [9]	train-auc:0.989222	valid-auc:0.978827
    
    F-Score:   0.80  previous:   0.83  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0331  threshold: 0.7239  random number: 0.9393 -> rejected
    
    Iteration =    13  T =     0.122825
    selected in next_choice: scale_pos_weight: from  50.00 to  40.00
    Parameters:
    eta :  0.2  subsample :  0.9  gamma :  0.15  scale_pos_weight :  40  max_depth :  5  colsample_bytree :  0.6  
    
    [0]	train-auc:0.946118	valid-auc:0.943968
    [9]	train-auc:0.989917	valid-auc:0.983616
    
    F-Score:   0.82  previous:   0.83  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0134  threshold: 0.8771  random number: 0.5822 -> accepted
    
    Iteration =    14  T =     0.122825
    selected in next_choice: max_depth : from   5.00 to  10.00
    Parameters:
    eta :  0.2  subsample :  0.9  gamma :  0.15  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.9726	valid-auc:0.931686
    [9]	train-auc:0.999963	valid-auc:0.9647
    
    F-Score:   0.82  previous:   0.82  best so far:   0.83
    Local improvement
    
    Iteration =    15  T =     0.122825
    selected in next_choice: colsample_bytree: from   0.60 to   0.70
    Parameters:
    eta :  0.2  subsample :  0.9  gamma :  0.15  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.7  
    
    [0]	train-auc:0.972601	valid-auc:0.931571
    [9]	train-auc:0.999955	valid-auc:0.965948
    
    F-Score:   0.79  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0325  threshold: 0.7279  random number: 0.6716 -> accepted
    
    Iteration =    16  T =     0.104401
    selected in next_choice: scale_pos_weight: from  40.00 to  30.00
    Parameters:
    eta :  0.2  subsample :  0.9  gamma :  0.15  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.7  
    
    [0]	train-auc:0.967464	valid-auc:0.938166
    [9]	train-auc:0.999975	valid-auc:0.964927
    
    F-Score:   0.80  previous:   0.79  best so far:   0.83
    Local improvement
    
    Iteration =    17  T =     0.104401
    selected in next_choice: scale_pos_weight: from  30.00 to  40.00
    
    Combination revisited - searching again
    selected in next_choice: colsample_bytree: from   0.70 to   0.60
    
    Combination revisited - searching again
    selected in next_choice: gamma     : from   0.15 to   0.10
    Parameters:
    eta :  0.2  subsample :  0.9  gamma :  0.1  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.9726	valid-auc:0.931686
    [9]	train-auc:0.999963	valid-auc:0.9647
    
    F-Score:   0.82  previous:   0.80  best so far:   0.83
    Local improvement
    
    Iteration =    18  T =     0.104401
    selected in next_choice: eta       : from   0.20 to   0.15
    Parameters:
    eta :  0.15  subsample :  0.9  gamma :  0.1  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.9726	valid-auc:0.931686
    [9]	train-auc:0.999854	valid-auc:0.968506
    
    F-Score:   0.82  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.0839 -> accepted
    
    Iteration =    19  T =     0.104401
    selected in next_choice: eta       : from   0.15 to   0.10
    Parameters:
    eta :  0.1  subsample :  0.9  gamma :  0.1  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.9726	valid-auc:0.931686
    [9]	train-auc:0.99971	valid-auc:0.964972
    
    F-Score:   0.81  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0087  threshold: 0.9049  random number: 0.7665 -> accepted
    
    Iteration =    20  T =     0.104401
    selected in next_choice: scale_pos_weight: from  40.00 to  30.00
    Parameters:
    eta :  0.1  subsample :  0.9  gamma :  0.1  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.967447	valid-auc:0.934786
    [9]	train-auc:0.999571	valid-auc:0.971973
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    21  T =     0.088741
    selected in next_choice: subsample : from   0.90 to   0.80
    Parameters:
    eta :  0.1  subsample :  0.8  gamma :  0.1  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.969693	valid-auc:0.944939
    [9]	train-auc:0.999879	valid-auc:0.978308
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    22  T =     0.088741
    selected in next_choice: gamma     : from   0.10 to   0.05
    Parameters:
    eta :  0.1  subsample :  0.8  gamma :  0.05  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.969693	valid-auc:0.944939
    [9]	train-auc:0.999879	valid-auc:0.978308
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.2368 -> accepted
    
    Iteration =    23  T =     0.088741
    selected in next_choice: subsample : from   0.80 to   0.90
    Parameters:
    eta :  0.1  subsample :  0.9  gamma :  0.05  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.967447	valid-auc:0.934786
    [9]	train-auc:0.999567	valid-auc:0.971452
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0027  threshold: 0.9638  random number: 0.0308 -> accepted
    
    Iteration =    24  T =     0.088741
    selected in next_choice: scale_pos_weight: from  30.00 to  40.00
    Parameters:
    eta :  0.1  subsample :  0.9  gamma :  0.05  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.9726	valid-auc:0.931686
    [9]	train-auc:0.999704	valid-auc:0.965264
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0002  threshold: 0.9979  random number: 0.7888 -> accepted
    
    Iteration =    25  T =     0.088741
    selected in next_choice: gamma     : from   0.05 to   0.10
    
    Combination revisited - searching again
    selected in next_choice: subsample : from   0.90 to   0.80
    Parameters:
    eta :  0.1  subsample :  0.8  gamma :  0.1  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.985898	valid-auc:0.953568
    [9]	train-auc:0.999905	valid-auc:0.969679
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    26  T =     0.075430
    selected in next_choice: eta       : from   0.10 to   0.15
    Parameters:
    eta :  0.15  subsample :  0.8  gamma :  0.1  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.985898	valid-auc:0.953568
    [9]	train-auc:0.999916	valid-auc:0.970392
    
    F-Score:   0.82  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    27  T =     0.075430
    selected in next_choice: max_depth : from  10.00 to  15.00
    Parameters:
    eta :  0.15  subsample :  0.8  gamma :  0.1  scale_pos_weight :  40  max_depth :  15  colsample_bytree :  0.6  
    
    [0]	train-auc:0.982413	valid-auc:0.930599
    [9]	train-auc:0.999919	valid-auc:0.969938
    
    F-Score:   0.82  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0043  threshold: 0.9339  random number: 0.3461 -> accepted
    
    Iteration =    28  T =     0.075430
    selected in next_choice: eta       : from   0.15 to   0.20
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.1  scale_pos_weight :  40  max_depth :  15  colsample_bytree :  0.6  
    
    [0]	train-auc:0.982413	valid-auc:0.930599
    [9]	train-auc:0.999938	valid-auc:0.961945
    
    F-Score:   0.82  previous:   0.82  best so far:   0.83
    Local improvement
    
    Iteration =    29  T =     0.075430
    selected in next_choice: scale_pos_weight: from  40.00 to  50.00
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.1  scale_pos_weight :  50  max_depth :  15  colsample_bytree :  0.6  
    
    [0]	train-auc:0.982401	valid-auc:0.930592
    [9]	train-auc:0.999917	valid-auc:0.964394
    
    F-Score:   0.82  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0056  threshold: 0.9150  random number: 0.6233 -> accepted
    
    Iteration =    30  T =     0.075430
    selected in next_choice: eta       : from   0.20 to   0.15
    Parameters:
    eta :  0.15  subsample :  0.8  gamma :  0.1  scale_pos_weight :  50  max_depth :  15  colsample_bytree :  0.6  
    
    [0]	train-auc:0.982401	valid-auc:0.930592
    [9]	train-auc:0.999919	valid-auc:0.96392
    
    F-Score:   0.82  previous:   0.82  best so far:   0.83
    Local improvement
    
    Iteration =    31  T =     0.064115
    selected in next_choice: scale_pos_weight: from  50.00 to 300.00
    Parameters:
    eta :  0.15  subsample :  0.8  gamma :  0.1  scale_pos_weight :  300  max_depth :  15  colsample_bytree :  0.6  
    
    [0]	train-auc:0.993286	valid-auc:0.979525
    [9]	train-auc:0.999926	valid-auc:0.979007
    
    F-Score:   0.81  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0190  threshold: 0.7013  random number: 0.6158 -> accepted
    
    Iteration =    32  T =     0.064115
    selected in next_choice: colsample_bytree: from   0.60 to   0.70
    Parameters:
    eta :  0.15  subsample :  0.8  gamma :  0.1  scale_pos_weight :  300  max_depth :  15  colsample_bytree :  0.7  
    
    [0]	train-auc:0.993451	valid-auc:0.978103
    [9]	train-auc:0.999918	valid-auc:0.978845
    
    F-Score:   0.82  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    33  T =     0.064115
    selected in next_choice: subsample : from   0.80 to   0.70
    Parameters:
    eta :  0.15  subsample :  0.7  gamma :  0.1  scale_pos_weight :  300  max_depth :  15  colsample_bytree :  0.7  
    
    [0]	train-auc:0.982521	valid-auc:0.976107
    [9]	train-auc:0.99991	valid-auc:0.977231
    
    F-Score:   0.81  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0145  threshold: 0.7617  random number: 0.1486 -> accepted
    
    Iteration =    34  T =     0.064115
    selected in next_choice: scale_pos_weight: from 300.00 to 400.00
    Parameters:
    eta :  0.15  subsample :  0.7  gamma :  0.1  scale_pos_weight :  400  max_depth :  15  colsample_bytree :  0.7  
    
    [0]	train-auc:0.98279	valid-auc:0.976332
    [9]	train-auc:0.999923	valid-auc:0.980894
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    35  T =     0.064115
    selected in next_choice: eta       : from   0.15 to   0.10
    Parameters:
    eta :  0.1  subsample :  0.7  gamma :  0.1  scale_pos_weight :  400  max_depth :  15  colsample_bytree :  0.7  
    
    [0]	train-auc:0.98279	valid-auc:0.976332
    [9]	train-auc:0.99991	valid-auc:0.97903
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    36  T =     0.054498
    selected in next_choice: eta       : from   0.10 to   0.15
    
    Combination revisited - searching again
    selected in next_choice: eta       : from   0.15 to   0.20
    Parameters:
    eta :  0.2  subsample :  0.7  gamma :  0.1  scale_pos_weight :  400  max_depth :  15  colsample_bytree :  0.7  
    
    [0]	train-auc:0.98279	valid-auc:0.976332
    [9]	train-auc:0.999909	valid-auc:0.969513
    
    F-Score:   0.80  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0103  threshold: 0.7977  random number: 0.1831 -> accepted
    
    Iteration =    37  T =     0.054498
    selected in next_choice: max_depth : from  15.00 to  10.00
    Parameters:
    eta :  0.2  subsample :  0.7  gamma :  0.1  scale_pos_weight :  400  max_depth :  10  colsample_bytree :  0.7  
    
    [0]	train-auc:0.982772	valid-auc:0.976302
    [9]	train-auc:0.999908	valid-auc:0.960039
    
    F-Score:   0.76  previous:   0.80  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0469  threshold: 0.3560  random number: 0.1144 -> accepted
    
    Iteration =    38  T =     0.054498
    selected in next_choice: max_depth : from  10.00 to   5.00
    Parameters:
    eta :  0.2  subsample :  0.7  gamma :  0.1  scale_pos_weight :  400  max_depth :  5  colsample_bytree :  0.7  
    
    [0]	train-auc:0.981663	valid-auc:0.975719
    [9]	train-auc:0.999729	valid-auc:0.976098
    
    F-Score:   0.51  previous:   0.76  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.2508  threshold: 0.0040  random number: 0.0146 -> rejected
    
    Iteration =    39  T =     0.054498
    selected in next_choice: subsample : from   0.70 to   0.80
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.1  scale_pos_weight :  400  max_depth :  5  colsample_bytree :  0.7  
    
    [0]	train-auc:0.976032	valid-auc:0.899881
    [9]	train-auc:0.999577	valid-auc:0.972076
    
    F-Score:   0.48  previous:   0.76  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.2751  threshold: 0.0023  random number: 0.4868 -> rejected
    
    Iteration =    40  T =     0.054498
    selected in next_choice: max_depth : from   5.00 to  10.00
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.1  scale_pos_weight :  400  max_depth :  10  colsample_bytree :  0.7  
    
    [0]	train-auc:0.993238	valid-auc:0.974637
    [9]	train-auc:0.999915	valid-auc:0.950112
    
    F-Score:   0.78  previous:   0.76  best so far:   0.83
    Local improvement
    
    Iteration =    41  T =     0.046323
    selected in next_choice: max_depth : from  10.00 to   5.00
    
    Combination revisited - searching again
    selected in next_choice: subsample : from   0.80 to   0.90
    Parameters:
    eta :  0.2  subsample :  0.9  gamma :  0.1  scale_pos_weight :  400  max_depth :  5  colsample_bytree :  0.7  
    
    [0]	train-auc:0.98214	valid-auc:0.901795
    [9]	train-auc:0.999761	valid-auc:0.980874
    
    F-Score:   0.50  previous:   0.78  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.2770  threshold: 0.0008  random number: 0.9649 -> rejected
    
    Iteration =    42  T =     0.046323
    selected in next_choice: gamma     : from   0.10 to   0.15
    Parameters:
    eta :  0.2  subsample :  0.9  gamma :  0.15  scale_pos_weight :  400  max_depth :  5  colsample_bytree :  0.7  
    
    [0]	train-auc:0.98214	valid-auc:0.901795
    [9]	train-auc:0.999761	valid-auc:0.980874
    
    F-Score:   0.50  previous:   0.78  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.2770  threshold: 0.0008  random number: 0.0646 -> rejected
    
    Iteration =    43  T =     0.046323
    selected in next_choice: eta       : from   0.20 to   0.15
    Parameters:
    eta :  0.15  subsample :  0.9  gamma :  0.15  scale_pos_weight :  400  max_depth :  5  colsample_bytree :  0.7  
    
    [0]	train-auc:0.98214	valid-auc:0.901795
    [9]	train-auc:0.999539	valid-auc:0.976451
    
    F-Score:   0.44  previous:   0.78  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.3355  threshold: 0.0002  random number: 0.5411 -> rejected
    
    Iteration =    44  T =     0.046323
    selected in next_choice: scale_pos_weight: from 400.00 to 300.00
    Parameters:
    eta :  0.15  subsample :  0.9  gamma :  0.15  scale_pos_weight :  300  max_depth :  5  colsample_bytree :  0.7  
    
    [0]	train-auc:0.982763	valid-auc:0.924169
    [9]	train-auc:0.999349	valid-auc:0.979659
    
    F-Score:   0.61  previous:   0.78  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.1668  threshold: 0.0133  random number: 0.4659 -> rejected
    
    Iteration =    45  T =     0.046323
    selected in next_choice: colsample_bytree: from   0.70 to   0.80
    Parameters:
    eta :  0.15  subsample :  0.9  gamma :  0.15  scale_pos_weight :  300  max_depth :  5  colsample_bytree :  0.8  
    
    [0]	train-auc:0.980586	valid-auc:0.90959
    [9]	train-auc:0.999699	valid-auc:0.965681
    
    F-Score:   0.66  previous:   0.78  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.1141  threshold: 0.0520  random number: 0.6015 -> rejected
    
    Iteration =    46  T =     0.039375
    selected in next_choice: scale_pos_weight: from 300.00 to  50.00
    Parameters:
    eta :  0.15  subsample :  0.9  gamma :  0.15  scale_pos_weight :  50  max_depth :  5  colsample_bytree :  0.8  
    
    [0]	train-auc:0.946664	valid-auc:0.923623
    [9]	train-auc:0.988132	valid-auc:0.98001
    
    F-Score:   0.77  previous:   0.78  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0119  threshold: 0.6953  random number: 0.0889 -> accepted
    
    Iteration =    47  T =     0.039375
    selected in next_choice: max_depth : from   5.00 to  10.00
    
    Combination revisited - searching again
    selected in next_choice: subsample : from   0.90 to   1.00
    Parameters:
    eta :  0.15  subsample :  1.0  gamma :  0.15  scale_pos_weight :  50  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.979527	valid-auc:0.940509
    [9]	train-auc:0.999695	valid-auc:0.969905
    
    F-Score:   0.76  previous:   0.77  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0058  threshold: 0.8385  random number: 0.5790 -> accepted
    
    Iteration =    48  T =     0.039375
    selected in next_choice: max_depth : from  10.00 to  15.00
    Parameters:
    eta :  0.15  subsample :  1.0  gamma :  0.15  scale_pos_weight :  50  max_depth :  15  colsample_bytree :  0.8  
    
    [0]	train-auc:0.999565	valid-auc:0.960823
    [9]	train-auc:0.999912	valid-auc:0.958344
    
    F-Score:   0.75  previous:   0.76  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0076  threshold: 0.7922  random number: 0.2696 -> accepted
    
    Iteration =    49  T =     0.039375
    selected in next_choice: max_depth : from  15.00 to  10.00
    
    Combination revisited - searching again
    selected in next_choice: scale_pos_weight: from  50.00 to  40.00
    Parameters:
    eta :  0.15  subsample :  1.0  gamma :  0.15  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.97953	valid-auc:0.933839
    [9]	train-auc:0.999645	valid-auc:0.964187
    
    F-Score:   0.76  previous:   0.75  best so far:   0.83
    Local improvement
    
    Iteration =    50  T =     0.039375
    selected in next_choice: scale_pos_weight: from  40.00 to  30.00
    Parameters:
    eta :  0.15  subsample :  1.0  gamma :  0.15  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.979543	valid-auc:0.933969
    [9]	train-auc:0.999768	valid-auc:0.973134
    
    F-Score:   0.77  previous:   0.76  best so far:   0.83
    Local improvement
    
    Iteration =    51  T =     0.033469
    selected in next_choice: gamma     : from   0.15 to   0.20
    Parameters:
    eta :  0.15  subsample :  1.0  gamma :  0.2  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.979543	valid-auc:0.933969
    [9]	train-auc:0.999768	valid-auc:0.973134
    
    F-Score:   0.77  previous:   0.77  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.5564 -> accepted
    
    Iteration =    52  T =     0.033469
    selected in next_choice: colsample_bytree: from   0.80 to   0.70
    Parameters:
    eta :  0.15  subsample :  1.0  gamma :  0.2  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.7  
    
    [0]	train-auc:0.988122	valid-auc:0.932805
    [9]	train-auc:0.999955	valid-auc:0.966969
    
    F-Score:   0.78  previous:   0.77  best so far:   0.83
    Local improvement
    
    Iteration =    53  T =     0.033469
    selected in next_choice: gamma     : from   0.20 to   0.15
    Parameters:
    eta :  0.15  subsample :  1.0  gamma :  0.15  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.7  
    
    [0]	train-auc:0.988122	valid-auc:0.932805
    [9]	train-auc:0.999955	valid-auc:0.966968
    
    F-Score:   0.78  previous:   0.78  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.6446 -> accepted
    
    Iteration =    54  T =     0.033469
    selected in next_choice: subsample : from   1.00 to   0.90
    Parameters:
    eta :  0.15  subsample :  0.9  gamma :  0.15  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.7  
    
    [0]	train-auc:0.967464	valid-auc:0.938166
    [9]	train-auc:0.99942	valid-auc:0.964276
    
    F-Score:   0.80  previous:   0.78  best so far:   0.83
    Local improvement
    
    Iteration =    55  T =     0.033469
    selected in next_choice: subsample : from   0.90 to   0.80
    Parameters:
    eta :  0.15  subsample :  0.8  gamma :  0.15  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.7  
    
    [0]	train-auc:0.969599	valid-auc:0.937706
    [9]	train-auc:0.999839	valid-auc:0.974833
    
    F-Score:   0.81  previous:   0.80  best so far:   0.83
    Local improvement
    
    Iteration =    56  T =     0.028448
    selected in next_choice: colsample_bytree: from   0.70 to   0.60
    Parameters:
    eta :  0.15  subsample :  0.8  gamma :  0.15  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.969693	valid-auc:0.944939
    [9]	train-auc:0.999907	valid-auc:0.9767
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    57  T =     0.028448
    selected in next_choice: gamma     : from   0.15 to   0.10
    Parameters:
    eta :  0.15  subsample :  0.8  gamma :  0.1  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.969693	valid-auc:0.944939
    [9]	train-auc:0.999907	valid-auc:0.9767
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.4810 -> accepted
    
    Iteration =    58  T =     0.028448
    selected in next_choice: max_depth : from  10.00 to   5.00
    Parameters:
    eta :  0.15  subsample :  0.8  gamma :  0.1  scale_pos_weight :  30  max_depth :  5  colsample_bytree :  0.6  
    
    [0]	train-auc:0.94184	valid-auc:0.934135
    [9]	train-auc:0.974279	valid-auc:0.955128
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    59  T =     0.028448
    selected in next_choice: gamma     : from   0.10 to   0.05
    Parameters:
    eta :  0.15  subsample :  0.8  gamma :  0.05  scale_pos_weight :  30  max_depth :  5  colsample_bytree :  0.6  
    
    [0]	train-auc:0.94184	valid-auc:0.934135
    [9]	train-auc:0.974283	valid-auc:0.955139
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.3552 -> accepted
    
    Iteration =    60  T =     0.028448
    selected in next_choice: eta       : from   0.15 to   0.20
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.05  scale_pos_weight :  30  max_depth :  5  colsample_bytree :  0.6  
    
    [0]	train-auc:0.94184	valid-auc:0.934135
    [9]	train-auc:0.988248	valid-auc:0.97642
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    61  T =     0.024181
    selected in next_choice: scale_pos_weight: from  30.00 to  40.00
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.05  scale_pos_weight :  40  max_depth :  5  colsample_bytree :  0.6  
    
    [0]	train-auc:0.957183	valid-auc:0.948786
    [9]	train-auc:0.984963	valid-auc:0.975702
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0031  threshold: 0.8575  random number: 0.2492 -> accepted
    
    Iteration =    62  T =     0.024181
    selected in next_choice: gamma     : from   0.05 to   0.10
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.1  scale_pos_weight :  40  max_depth :  5  colsample_bytree :  0.6  
    
    [0]	train-auc:0.957183	valid-auc:0.948786
    [9]	train-auc:0.984963	valid-auc:0.975702
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.9335 -> accepted
    
    Iteration =    63  T =     0.024181
    selected in next_choice: max_depth : from   5.00 to  10.00
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.1  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.985898	valid-auc:0.953568
    [9]	train-auc:0.999937	valid-auc:0.953897
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0012  threshold: 0.9444  random number: 0.4534 -> accepted
    
    Iteration =    64  T =     0.024181
    selected in next_choice: gamma     : from   0.10 to   0.05
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.05  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.985898	valid-auc:0.953568
    [9]	train-auc:0.999937	valid-auc:0.953897
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.5302 -> accepted
    
    Iteration =    65  T =     0.024181
    selected in next_choice: scale_pos_weight: from  40.00 to  30.00
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.05  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.6  
    
    [0]	train-auc:0.969693	valid-auc:0.944939
    [9]	train-auc:0.999969	valid-auc:0.975821
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0029  threshold: 0.8663  random number: 0.0193 -> accepted
    
    Iteration =    66  T =     0.020554
    selected in next_choice: colsample_bytree: from   0.60 to   0.70
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.05  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.7  
    
    [0]	train-auc:0.969599	valid-auc:0.937706
    [9]	train-auc:0.999965	valid-auc:0.980782
    
    F-Score:   0.80  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0043  threshold: 0.7768  random number: 0.5081 -> accepted
    
    Iteration =    67  T =     0.020554
    selected in next_choice: max_depth : from  10.00 to   5.00
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.05  scale_pos_weight :  30  max_depth :  5  colsample_bytree :  0.7  
    
    [0]	train-auc:0.941746	valid-auc:0.93411
    [9]	train-auc:0.984556	valid-auc:0.967545
    
    F-Score:   0.79  previous:   0.80  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0124  threshold: 0.4837  random number: 0.0058 -> accepted
    
    Iteration =    68  T =     0.020554
    selected in next_choice: colsample_bytree: from   0.70 to   0.80
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.05  scale_pos_weight :  30  max_depth :  5  colsample_bytree :  0.8  
    
    [0]	train-auc:0.941746	valid-auc:0.93411
    [9]	train-auc:0.991635	valid-auc:0.982195
    
    F-Score:   0.81  previous:   0.79  best so far:   0.83
    Local improvement
    
    Iteration =    69  T =     0.020554
    selected in next_choice: scale_pos_weight: from  30.00 to  40.00
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.05  scale_pos_weight :  40  max_depth :  5  colsample_bytree :  0.8  
    
    [0]	train-auc:0.956901	valid-auc:0.948084
    [9]	train-auc:0.989147	valid-auc:0.983087
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0016  threshold: 0.9129  random number: 0.1438 -> accepted
    
    Iteration =    70  T =     0.020554
    selected in next_choice: max_depth : from   5.00 to  10.00
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.05  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.982146	valid-auc:0.936521
    [9]	train-auc:0.999957	valid-auc:0.966623
    
    F-Score:   0.80  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0072  threshold: 0.6571  random number: 0.4728 -> accepted
    
    Iteration =    71  T =     0.017471
    selected in next_choice: scale_pos_weight: from  40.00 to  30.00
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.05  scale_pos_weight :  30  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.969599	valid-auc:0.937706
    [9]	train-auc:0.99995	valid-auc:0.973748
    
    F-Score:   0.79  previous:   0.80  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0125  threshold: 0.4229  random number: 0.3773 -> accepted
    
    Iteration =    72  T =     0.017471
    selected in next_choice: scale_pos_weight: from  30.00 to  40.00
    
    Combination revisited - searching again
    selected in next_choice: gamma     : from   0.05 to   0.10
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.1  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.8  
    
    [0]	train-auc:0.982146	valid-auc:0.936521
    [9]	train-auc:0.999957	valid-auc:0.966623
    
    F-Score:   0.80  previous:   0.79  best so far:   0.83
    Local improvement
    
    Iteration =    73  T =     0.017471
    selected in next_choice: max_depth : from  10.00 to   5.00
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.1  scale_pos_weight :  40  max_depth :  5  colsample_bytree :  0.8  
    
    [0]	train-auc:0.956901	valid-auc:0.948084
    [9]	train-auc:0.989147	valid-auc:0.983087
    
    F-Score:   0.81  previous:   0.80  best so far:   0.83
    Local improvement
    
    Iteration =    74  T =     0.017471
    selected in next_choice: subsample : from   0.80 to   0.70
    Parameters:
    eta :  0.2  subsample :  0.7  gamma :  0.1  scale_pos_weight :  40  max_depth :  5  colsample_bytree :  0.8  
    
    [0]	train-auc:0.957335	valid-auc:0.948866
    [9]	train-auc:0.992334	valid-auc:0.976264
    
    F-Score:   0.82  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    75  T =     0.017471
    selected in next_choice: scale_pos_weight: from  40.00 to  30.00
    Parameters:
    eta :  0.2  subsample :  0.7  gamma :  0.1  scale_pos_weight :  30  max_depth :  5  colsample_bytree :  0.8  
    
    [0]	train-auc:0.941835	valid-auc:0.934143
    [9]	train-auc:0.988599	valid-auc:0.975537
    
    F-Score:   0.83  previous:   0.82  best so far:   0.83
    Local improvement
    
    Iteration =    76  T =     0.014850
    selected in next_choice: scale_pos_weight: from  30.00 to  40.00
    
    Combination revisited - searching again
    selected in next_choice: colsample_bytree: from   0.80 to   0.90
    Parameters:
    eta :  0.2  subsample :  0.7  gamma :  0.1  scale_pos_weight :  40  max_depth :  5  colsample_bytree :  0.9  
    
    [0]	train-auc:0.937844	valid-auc:0.931795
    [9]	train-auc:0.989042	valid-auc:0.976264
    
    F-Score:   0.81  previous:   0.83  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0143  threshold: 0.3153  random number: 0.0542 -> accepted
    
    Iteration =    77  T =     0.014850
    selected in next_choice: max_depth : from   5.00 to  10.00
    Parameters:
    eta :  0.2  subsample :  0.7  gamma :  0.1  scale_pos_weight :  40  max_depth :  10  colsample_bytree :  0.9  
    
    [0]	train-auc:0.97994	valid-auc:0.966859
    [9]	train-auc:0.999958	valid-auc:0.968255
    
    F-Score:   0.80  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0129  threshold: 0.3535  random number: 0.5875 -> rejected
    
    Iteration =    78  T =     0.014850
    selected in next_choice: max_depth : from  10.00 to  15.00
    Parameters:
    eta :  0.2  subsample :  0.7  gamma :  0.1  scale_pos_weight :  40  max_depth :  15  colsample_bytree :  0.9  
    
    [0]	train-auc:0.980877	valid-auc:0.969225
    [9]	train-auc:0.999959	valid-auc:0.969989
    
    F-Score:   0.80  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0185  threshold: 0.2241  random number: 0.1640 -> accepted
    
    Iteration =    79  T =     0.014850
    selected in next_choice: scale_pos_weight: from  40.00 to  30.00
    Parameters:
    eta :  0.2  subsample :  0.7  gamma :  0.1  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.9  
    
    [0]	train-auc:0.980879	valid-auc:0.96929
    [9]	train-auc:0.999969	valid-auc:0.962677
    
    F-Score:   0.79  previous:   0.80  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0056  threshold: 0.6378  random number: 0.5573 -> accepted
    
    Iteration =    80  T =     0.014850
    selected in next_choice: subsample : from   0.70 to   0.80
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.1  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.9  
    
    [0]	train-auc:0.982268	valid-auc:0.944617
    [9]	train-auc:0.999951	valid-auc:0.947622
    
    F-Score:   0.78  previous:   0.79  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0124  threshold: 0.3662  random number: 0.1442 -> accepted
    
    Iteration =    81  T =     0.012623
    selected in next_choice: subsample : from   0.80 to   0.70
    
    Combination revisited - searching again
    selected in next_choice: colsample_bytree: from   0.90 to   0.80
    Parameters:
    eta :  0.2  subsample :  0.7  gamma :  0.1  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.8  
    
    [0]	train-auc:0.980951	valid-auc:0.964096
    [9]	train-auc:0.999955	valid-auc:0.967532
    
    F-Score:   0.81  previous:   0.78  best so far:   0.83
    Local improvement
    
    Iteration =    82  T =     0.012623
    selected in next_choice: eta       : from   0.20 to   0.25
    Parameters:
    eta :  0.25  subsample :  0.7  gamma :  0.1  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.8  
    
    [0]	train-auc:0.980951	valid-auc:0.964096
    [9]	train-auc:0.999978	valid-auc:0.966473
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0015  threshold: 0.8647  random number: 0.9373 -> rejected
    
    Iteration =    83  T =     0.012623
    selected in next_choice: gamma     : from   0.10 to   0.05
    Parameters:
    eta :  0.25  subsample :  0.7  gamma :  0.05  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.8  
    
    [0]	train-auc:0.980951	valid-auc:0.964096
    [9]	train-auc:0.999978	valid-auc:0.966473
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0015  threshold: 0.8647  random number: 0.7710 -> accepted
    
    Iteration =    84  T =     0.012623
    selected in next_choice: eta       : from   0.25 to   0.30
    Parameters:
    eta :  0.3  subsample :  0.7  gamma :  0.05  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.8  
    
    [0]	train-auc:0.980951	valid-auc:0.964096
    [9]	train-auc:0.999983	valid-auc:0.970061
    
    F-Score:   0.82  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    85  T =     0.012623
    selected in next_choice: gamma     : from   0.05 to   0.00
    Parameters:
    eta :  0.3  subsample :  0.7  gamma :  0.0  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.8  
    
    [0]	train-auc:0.980951	valid-auc:0.964096
    [9]	train-auc:0.999984	valid-auc:0.970061
    
    F-Score:   0.82  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.9569 -> accepted
    
    Iteration =    86  T =     0.010729
    selected in next_choice: subsample : from   0.70 to   0.60
    Parameters:
    eta :  0.3  subsample :  0.6  gamma :  0.0  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.8  
    
    [0]	train-auc:0.980171	valid-auc:0.963853
    [9]	train-auc:0.999969	valid-auc:0.976469
    
    F-Score:   0.81  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0027  threshold: 0.7428  random number: 0.1412 -> accepted
    
    Iteration =    87  T =     0.010729
    selected in next_choice: max_depth : from  15.00 to  20.00
    Parameters:
    eta :  0.3  subsample :  0.6  gamma :  0.0  scale_pos_weight :  30  max_depth :  20  colsample_bytree :  0.8  
    
    [0]	train-auc:0.980171	valid-auc:0.963853
    [9]	train-auc:0.999969	valid-auc:0.976469
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.3054 -> accepted
    
    Iteration =    88  T =     0.010729
    selected in next_choice: max_depth : from  20.00 to  15.00
    
    Combination revisited - searching again
    selected in next_choice: gamma     : from   0.00 to   0.05
    Parameters:
    eta :  0.3  subsample :  0.6  gamma :  0.05  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.8  
    
    [0]	train-auc:0.980171	valid-auc:0.963853
    [9]	train-auc:0.999969	valid-auc:0.976469
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.0396 -> accepted
    
    Iteration =    89  T =     0.010729
    selected in next_choice: colsample_bytree: from   0.80 to   0.70
    Parameters:
    eta :  0.3  subsample :  0.6  gamma :  0.05  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.7  
    
    [0]	train-auc:0.980163	valid-auc:0.96385
    [9]	train-auc:0.999975	valid-auc:0.977051
    
    F-Score:   0.82  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    90  T =     0.010729
    selected in next_choice: subsample : from   0.60 to   0.50
    Parameters:
    eta :  0.3  subsample :  0.5  gamma :  0.05  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.7  
    
    [0]	train-auc:0.975489	valid-auc:0.96254
    [9]	train-auc:0.996406	valid-auc:0.972038
    
    F-Score:   0.81  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0130  threshold: 0.2338  random number: 0.2768 -> rejected
    
    Iteration =    91  T =     0.009120
    selected in next_choice: colsample_bytree: from   0.70 to   0.60
    Parameters:
    eta :  0.3  subsample :  0.5  gamma :  0.05  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.6  
    
    [0]	train-auc:0.975489	valid-auc:0.96254
    [9]	train-auc:0.996289	valid-auc:0.967806
    
    F-Score:   0.82  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0056  threshold: 0.4760  random number: 0.8065 -> rejected
    
    Iteration =    92  T =     0.009120
    selected in next_choice: gamma     : from   0.05 to   0.00
    Parameters:
    eta :  0.3  subsample :  0.5  gamma :  0.0  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.6  
    
    [0]	train-auc:0.975489	valid-auc:0.96254
    [9]	train-auc:0.996289	valid-auc:0.967811
    
    F-Score:   0.82  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0056  threshold: 0.4760  random number: 0.1773 -> accepted
    
    Iteration =    93  T =     0.009120
    selected in next_choice: colsample_bytree: from   0.60 to   0.50
    Parameters:
    eta :  0.3  subsample :  0.5  gamma :  0.0  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.5  
    
    [0]	train-auc:0.97823	valid-auc:0.970655
    [9]	train-auc:0.998184	valid-auc:0.975574
    
    F-Score:   0.82  previous:   0.82  best so far:   0.83
    Local improvement
    
    Iteration =    94  T =     0.009120
    selected in next_choice: gamma     : from   0.00 to   0.05
    Parameters:
    eta :  0.3  subsample :  0.5  gamma :  0.05  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.5  
    
    [0]	train-auc:0.97823	valid-auc:0.970655
    [9]	train-auc:0.998211	valid-auc:0.974836
    
    F-Score:   0.81  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0030  threshold: 0.6763  random number: 0.1546 -> accepted
    
    Iteration =    95  T =     0.009120
    selected in next_choice: subsample : from   0.50 to   0.60
    Parameters:
    eta :  0.3  subsample :  0.6  gamma :  0.05  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.5  
    
    [0]	train-auc:0.982153	valid-auc:0.978917
    [9]	train-auc:0.999976	valid-auc:0.973996
    
    F-Score:   0.81  previous:   0.81  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.9547 -> accepted
    
    Iteration =    96  T =     0.007752
    selected in next_choice: scale_pos_weight: from  30.00 to  40.00
    Parameters:
    eta :  0.3  subsample :  0.6  gamma :  0.05  scale_pos_weight :  40  max_depth :  15  colsample_bytree :  0.5  
    
    [0]	train-auc:0.982148	valid-auc:0.978923
    [9]	train-auc:0.999974	valid-auc:0.979448
    
    F-Score:   0.82  previous:   0.81  best so far:   0.83
    Local improvement
    
    Iteration =    97  T =     0.007752
    selected in next_choice: gamma     : from   0.05 to   0.10
    Parameters:
    eta :  0.3  subsample :  0.6  gamma :  0.1  scale_pos_weight :  40  max_depth :  15  colsample_bytree :  0.5  
    
    [0]	train-auc:0.982148	valid-auc:0.978923
    [9]	train-auc:0.999974	valid-auc:0.979486
    
    F-Score:   0.82  previous:   0.82  best so far:   0.83
    Worse result
    
    F-Score decline:   0.0000  threshold: 1.0000  random number: 0.1546 -> accepted
    
    Iteration =    98  T =     0.007752
    selected in next_choice: colsample_bytree: from   0.50 to   0.60
    Parameters:
    eta :  0.3  subsample :  0.6  gamma :  0.1  scale_pos_weight :  40  max_depth :  15  colsample_bytree :  0.6  
    
    [0]	train-auc:0.982163	valid-auc:0.979002
    [9]	train-auc:0.999956	valid-auc:0.974888
    
    F-Score:   0.83  previous:   0.82  best so far:   0.83
    Local improvement
    
    Iteration =    99  T =     0.007752
    selected in next_choice: scale_pos_weight: from  40.00 to  30.00
    Parameters:
    eta :  0.3  subsample :  0.6  gamma :  0.1  scale_pos_weight :  30  max_depth :  15  colsample_bytree :  0.6  
    
    [0]	train-auc:0.980232	valid-auc:0.963372
    [9]	train-auc:0.999971	valid-auc:0.973973
    
    F-Score:   0.81  previous:   0.83  best so far:   0.83
    Worse result
    
    F-Score decline:  -0.0161  threshold: 0.0822  random number: 0.8339 -> rejected
    
     945.2413119014333  seconds process time
    
    Best variable parameters found:
    
    {'eta': 0.2, 'subsample': 0.8, 'gamma': 0.15, 'scale_pos_weight': 50, 'max_depth': 5, 'colsample_bytree': 0.6}
    

## Evaluation on the test dataset.

The evaluation on the test dataset results to an F-Score of 0.85 which is quite good. The plot showing the F-Score during the Simulated Annealing iterations clearly shows large fluctuations in the first iterations and much smaller in the last ones.

The best hyper-parameters found are in the ranges expected to be good according to all sources. Importantly, one can proceed this way:
* narrowing the ranges of these hyper-parameters   
* possibly adding others which are not used here (for example, regularization parameters)
* possibly doing some variable selection on the basis of the variable importance information.



```python
print('\nEvaluation on the test dataset\n')  

best_f_score,best_model=do_train(best_params, param, dtrain,'train',trainY,dtest,'test',testY,print_conf_matrix=True)


print('\nf-score on the test dataset: {:6.2f}'.format(best_f_score))

plt.plot(results['F-Score'])
plt.xlabel('Iterations')
plt.ylabel('F-Score')
plt.show()


xgb.plot_importance(best_model) 



```

    
    Evaluation on the test dataset
    
    Parameters:
    eta :  0.2  subsample :  0.8  gamma :  0.15  scale_pos_weight :  50  max_depth :  5  colsample_bytree :  0.6  
    
    [0]	train-auc:0.957356	test-auc:0.949645
    [9]	train-auc:0.989536	test-auc:0.976601
    
    confusion matrix
    ----------------
    tn: 85275 fp:    19
    fn:    29 tp:   119
    
    f-score on the test dataset:   0.83
    


![png](output_17_1.png)





    <matplotlib.axes._subplots.AxesSubplot at 0xbcdc748>




![png](output_17_3.png)


### Plot 


```python
best_params
```




    {'colsample_bytree': 0.6,
     'eta': 0.2,
     'gamma': 0.15,
     'max_depth': 5,
     'scale_pos_weight': 50,
     'subsample': 0.8}




```python
results['Duplicate']= results.duplicated([*tune_dic.keys()],keep='first')
results

##results['F-Score'][results['F-Score'] <= 0.60]

{k: param[k] for k in tune_dic.keys()}

results
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_depth</th>
      <th>subsample</th>
      <th>colsample_bytree</th>
      <th>eta</th>
      <th>gamma</th>
      <th>scale_pos_weight</th>
      <th>F-Score</th>
      <th>Best F-Score</th>
      <th>Duplicate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.15</td>
      <td>0.2</td>
      <td>50</td>
      <td>0.772881</td>
      <td>0.772881</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.15</td>
      <td>0.2</td>
      <td>40</td>
      <td>0.77931</td>
      <td>0.77931</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.15</td>
      <td>0.15</td>
      <td>50</td>
      <td>0.772881</td>
      <td>0.77931</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.15</td>
      <td>0.15</td>
      <td>300</td>
      <td>0.742671</td>
      <td>0.77931</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>0.15</td>
      <td>0.15</td>
      <td>300</td>
      <td>0.721311</td>
      <td>0.77931</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.15</td>
      <td>300</td>
      <td>0.753333</td>
      <td>0.77931</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.25</td>
      <td>0.15</td>
      <td>300</td>
      <td>0.765101</td>
      <td>0.77931</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.15</td>
      <td>50</td>
      <td>0.784983</td>
      <td>0.784983</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.15</td>
      <td>50</td>
      <td>0.767677</td>
      <td>0.784983</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.15</td>
      <td>50</td>
      <td>0.805654</td>
      <td>0.805654</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5</td>
      <td>0.8</td>
      <td>0.7</td>
      <td>0.2</td>
      <td>0.15</td>
      <td>50</td>
      <td>0.819788</td>
      <td>0.819788</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.15</td>
      <td>50</td>
      <td>0.830325</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.15</td>
      <td>50</td>
      <td>0.797251</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.15</td>
      <td>40</td>
      <td>0.816901</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.15</td>
      <td>40</td>
      <td>0.821429</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.7</td>
      <td>0.2</td>
      <td>0.15</td>
      <td>40</td>
      <td>0.788927</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.7</td>
      <td>0.2</td>
      <td>0.15</td>
      <td>30</td>
      <td>0.801418</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.817204</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.15</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.817204</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.808511</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>30</td>
      <td>0.808664</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>10</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>30</td>
      <td>0.811388</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>10</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.1</td>
      <td>0.05</td>
      <td>30</td>
      <td>0.811388</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.1</td>
      <td>0.05</td>
      <td>30</td>
      <td>0.808664</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>10</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.1</td>
      <td>0.05</td>
      <td>40</td>
      <td>0.808511</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>10</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.81295</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.15</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.821818</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>15</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.15</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.817518</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>15</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.823105</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>15</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>50</td>
      <td>0.817518</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>10</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.05</td>
      <td>40</td>
      <td>0.804196</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>71</th>
      <td>10</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.05</td>
      <td>30</td>
      <td>0.791667</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>72</th>
      <td>10</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.804196</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>73</th>
      <td>5</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.811388</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>74</th>
      <td>5</td>
      <td>0.7</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.824373</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>75</th>
      <td>5</td>
      <td>0.7</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>30</td>
      <td>0.828571</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>76</th>
      <td>5</td>
      <td>0.7</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.814286</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>77</th>
      <td>10</td>
      <td>0.7</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.801418</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>78</th>
      <td>15</td>
      <td>0.7</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.795775</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>79</th>
      <td>15</td>
      <td>0.7</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>30</td>
      <td>0.79021</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>80</th>
      <td>15</td>
      <td>0.8</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>30</td>
      <td>0.777778</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>81</th>
      <td>15</td>
      <td>0.7</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>30</td>
      <td>0.811388</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>82</th>
      <td>15</td>
      <td>0.7</td>
      <td>0.8</td>
      <td>0.25</td>
      <td>0.1</td>
      <td>30</td>
      <td>0.809859</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>83</th>
      <td>15</td>
      <td>0.7</td>
      <td>0.8</td>
      <td>0.25</td>
      <td>0.05</td>
      <td>30</td>
      <td>0.809859</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>84</th>
      <td>15</td>
      <td>0.7</td>
      <td>0.8</td>
      <td>0.3</td>
      <td>0.05</td>
      <td>30</td>
      <td>0.817204</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>85</th>
      <td>15</td>
      <td>0.7</td>
      <td>0.8</td>
      <td>0.3</td>
      <td>0</td>
      <td>30</td>
      <td>0.817204</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>86</th>
      <td>15</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.3</td>
      <td>0</td>
      <td>30</td>
      <td>0.814545</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>87</th>
      <td>20</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.3</td>
      <td>0</td>
      <td>30</td>
      <td>0.814545</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>88</th>
      <td>15</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.3</td>
      <td>0.05</td>
      <td>30</td>
      <td>0.814545</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>89</th>
      <td>15</td>
      <td>0.6</td>
      <td>0.7</td>
      <td>0.3</td>
      <td>0.05</td>
      <td>30</td>
      <td>0.821818</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>90</th>
      <td>15</td>
      <td>0.5</td>
      <td>0.7</td>
      <td>0.3</td>
      <td>0.05</td>
      <td>30</td>
      <td>0.808824</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>91</th>
      <td>15</td>
      <td>0.5</td>
      <td>0.6</td>
      <td>0.3</td>
      <td>0.05</td>
      <td>30</td>
      <td>0.816176</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>92</th>
      <td>15</td>
      <td>0.5</td>
      <td>0.6</td>
      <td>0.3</td>
      <td>0</td>
      <td>30</td>
      <td>0.816176</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>93</th>
      <td>15</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>0</td>
      <td>30</td>
      <td>0.817518</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>94</th>
      <td>15</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>0.05</td>
      <td>30</td>
      <td>0.814545</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>95</th>
      <td>15</td>
      <td>0.6</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>0.05</td>
      <td>30</td>
      <td>0.814545</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>96</th>
      <td>15</td>
      <td>0.6</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>0.05</td>
      <td>40</td>
      <td>0.824818</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>97</th>
      <td>15</td>
      <td>0.6</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.824818</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>98</th>
      <td>15</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>40</td>
      <td>0.829091</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
    <tr>
      <th>99</th>
      <td>15</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>30</td>
      <td>0.81295</td>
      <td>0.830325</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 9 columns</p>
</div>


