import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy import signal
import sklearn as sk
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split # split into train/test dataset
from sklearn.preprocessing import StandardScaler   # for normalizing data
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score, precision_score, recall_score

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.6f}'.format


#1. LOAD DATASET in 'turbofan_data.csv' and print out description, info of the data
original_pd = pd.read_csv('turbofan_data.csv', sep=',')
print(original_pd.shape)  # number of rows, cols
print(original_pd.info())

#2a. REMOVE NULL columns: EGT, HP_MA
#2b. REMOVE CONSTANT columns:  P2, WF, EPR,  OS_MB, LP_MA, AUXD_BYPASS, EGT_RAWC, CAUXB_BF,FAN_DECOUP, P25_BLEED
#2c. REMOVE NEARLY CONSTANT columns: AUXA_BYPASS, AUXC_BYPASS, CAUXB, LPr_MA
# columns: CA and CAUXA(13k/20k), OS_MA(10k/20k) ==> have too many NaN value ==> REMOVE
# column BDIS is the negative value of column T2 ==> remove BDIS

## T2, GB_MA, GB_MB, OS_MC, STATIC_AUXA, AUXB_BYPASS  ==> LARGE values ==> REMOVE (strong bias)

remove_null = original_pd.drop(['EGT', 'CAUXB','HP_MA', 'P2', 'WF','EPR', 'OS_MB', 'LP_MA','AUXA_BYPASS','AUXC_BYPASS', 'AUXD_BYPASS', 'EGT_RAWC', 'CAUXB_BF', 'FAN_DECOUP', 'P25_BLEED'], axis=1)
remove_null = remove_null.drop(['CA', 'CAUXA', 'OS_MA', 'BDIS','LPr_MA'],axis=1)
remove_null = remove_null.drop(['GB_MA', 'GB_MB', 'OS_MC','STATIC_AUXA','AUXB_BYPASS','EGT_RAWC_SENSED'],axis=1)



groupby_ESN = remove_null.groupby('ESN')
#groupby_ESN.count()
#3.a   ADD DATE column starting from 1-Jan-2022  FOR EACH ESN (A1, A2... A99)

newdf_withDate = None
for name, group in groupby_ESN:
    group['date'] = pd.date_range(start ='1-Jan-2022', periods=len(group), freq='D')
    if newdf_withDate is None:
        newdf_withDate = group
    else:
        newdf_withDate= pd.concat([newdf_withDate, group], ignore_index=True)

#3.b DROP column time_(cycles)
newdf_withDate= newdf_withDate.drop(['time_(cycles)'],axis=1)


#4 ENGINE FAILED
# search for final cycel (i.e RUL = 0) and sort by 'date'.
# 100 engines (index 0-99) ==> cutoff date at position 33
finalCycle = newdf_withDate[newdf_withDate['RUL'] ==0] [['ESN','date']]
finalCycle= finalCycle.sort_values(by = ['date'])
finalCycle=finalCycle.reset_index(drop=True)

#CUTOFF DATE
print(finalCycle.iloc[33,1])
print("failed date beyond cutoff date: ", len(finalCycle[finalCycle['date']>= finalCycle.iloc[33,1]]))
print("failed date before cutoff date:",len(finalCycle[finalCycle['date']< finalCycle.iloc[33,1]]))
print(finalCycle.loc[0:33])
finalCycle.loc[0:33].to_csv('bad_cohort.csv',index=False)
## Filter by cuttoff date
filtered_data = newdf_withDate[newdf_withDate['date'] < finalCycle.iloc[33,1]]


#5 FILL MISSING VALUE with mean
# for i in filtered_data.columns[filtered_data.isnull().any(axis=0)]:    
#     filtered_data[i].fillna(filtered_data[i].mean(),inplace=True)

group_fd = filtered_data.groupby('ESN')
fillna_fd = None
for name, group in group_fd:
    for i in range(len(group.columns)-1): 
        for r in range(len(group)-1):
            if pd.isna(group.iloc[r,i]):
                group.iloc[r,i]= (group.iloc[r-1,i]+group.iloc[r+1,i])/2
    if fillna_fd is None:
        fillna_fd = group
    else:
        fillna_fd= pd.concat([fillna_fd, group], ignore_index=True)

#write this data set to .csv 
fillna_fd.to_csv('early_failed_engines.csv',index=False)
# test= pd.read_csv('early_failed_engines.csv')
# test


#6.a SavgolTransformer class

class SavgolTransformer:
    def __init__(self,device_col, window_length, polyorder, deriv):
        self.device_col = device_col
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
    
    
    def transform (self,data):
        grouped = data.groupby(self.device_col)  # group by device_col
        smoothed_df = None
        for name, group in grouped:
            group['GP_MA'] = signal.savgol_filter(group['GP_MA'], window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv)
            group['HC_MA'] = signal.savgol_filter(group['HC_MA'], window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv)
            group['T2'] = signal.savgol_filter(group['T2'], window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv)
            group['EGT_RAW'] = signal.savgol_filter(group['EGT_RAW'], window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv)
            group['VIBS'] = signal.savgol_filter(group['VIBS'], window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv)
            group['GS_VIBS'] = signal.savgol_filter(group['GS_VIBS'], window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv)
            if smoothed_df is None:
                smoothed_df =  group
            else:
                smoothed_df= pd.concat([smoothed_df, group], ignore_index=True)
        return smoothed_df


#6.b ########### SMOOTHED DATA using SavgolTransformer ###########
w= 11
p = 3
d= 0
savgol = SavgolTransformer(device_col = "ESN", window_length = w, polyorder = p, deriv = d)
smoothed_data = savgol.transform(fillna_fd)


########### MODELLING ##############
group_smd = smoothed_data.groupby('ESN')
flatten_df = None
for name, group in group_smd:
    df = pd.DataFrame.from_dict({'ESN':[group.tail(15).iloc[0,0]], 
         'GP_MA': [group.tail(15)['GP_MA'].mean()],
         'HC_MA': [group.tail(15)['HC_MA'].mean()],
         'T2': [group.tail(15)['T2'].mean()],
         'EGT_RAW': [group.tail(15)['EGT_RAW'].mean()],
         'VIBS': [group.tail(15)['VIBS'].mean()],
         'GS_VIBS': [group.tail(15)['GS_VIBS'].mean()] })
    if flatten_df is None:
       flatten_df= df
    else:
        flatten_df = pd.concat([flatten_df, df], ignore_index=True)

# ADD 'bad_cohort' LABEL
flatten_df['bad_cohort'] = flatten_df.apply(lambda x: 1 if x.iloc[0] in finalCycle.loc[0:33]['ESN'].values else 0 ,axis=1)
flatten_df
# We use data (mean) of engines (100 engines) to predict failure status  (bad_cohort): 1 ==> Failed, 0 ==> Runing
# This is a BINARY CLASSIFICATION problem. ==> we use MLPClassifier as it popular for 'common' classification problem and return high accuracy result
#################################################
import sklearn as sk
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split # split into train/test dataset
from sklearn.preprocessing import StandardScaler   # for normalizing data
from sklearn.preprocessing import MinMaxScaler # for normalizing data
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import warnings

X = flatten_df[['GP_MA','HC_MA','T2', 'EGT_RAW','VIBS','GS_VIBS']]
y = flatten_df['bad_cohort']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.20, random_state=0)


#ss_train = StandardScaler()
ss_train = MinMaxScaler()
X_train = ss_train.fit_transform(X_train)
#ss_test = StandardScaler()
ss_test = MinMaxScaler()
X_test = ss_test.fit_transform(X_test)

models,accuracy, precision, recall ,cm = {}, {}, {} ,{},{}


######## params
learning_rate= 0.08
alpha =0.5
max_iter = 2000
random_state = 1

params = [
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "momentum": 0,
        "learning_rate_init": learning_rate,
    },
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "learning_rate_init": learning_rate,
    },

    {"solver": "lbfgs","learning_rate_init": learning_rate},
    {"solver": "adam", "learning_rate_init": learning_rate}
]

models['MLPClassifier 1'] = MLPClassifier(random_state=random_state, alpha= alpha, max_iter=max_iter, **params[0])
models['MLPClassifier 2'] = MLPClassifier(random_state=random_state, alpha= alpha, max_iter=max_iter, **params[1])
models['MLPClassifier 3'] = MLPClassifier(random_state=random_state, alpha= alpha, max_iter=max_iter, **params[2])
models['MLPClassifier 4'] = MLPClassifier(random_state=random_state, alpha= alpha, max_iter=max_iter, **params[3])

for key in models.keys():
    with warnings.catch_warnings():
        warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module="sklearn"
            )
        # Fit the classifier
        models[key].fit(X_train, y_train)

    # Make predictions
    predictions = models[key].predict(X_test)

    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)
    cm[key] = confusion_matrix(y_test, predictions)

result = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
result['Accuracy'] = accuracy.values()
result['Precision'] = precision.values()
result['Recall'] = recall.values()
result['Confusion_Matrix']= cm.values()
print(result)


####### write ranking engines
ss_rank = MinMaxScaler()
X_ranking = ss_rank.fit_transform(X)
y_ranking = models['MLPClassifier 3'].predict_proba(X_ranking)
belong_to = y_ranking[:,0].tolist()
flatten_df['ranking_score']= belong_to
flatten_df= flatten_df.sort_values(by=['ranking_score'],ascending=False)
flatten_df['ranking']= range(1,len(flatten_df)+1)
flatten_df= flatten_df.reset_index(drop=True)
flatten_df[['ESN','ranking_score','ranking']].to_csv('ranking.csv',index=False)
    