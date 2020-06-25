import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


filepath='/home/pratishk28/spam.csv'
df=pd.read_csv(filepath, encoding='latin-1')


df=df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)
df=df.rename(columns={'v1':'labels','v2': 'sms'})


df['labels']=df.labels.map({'spam':0, 'ham':1})


count_vector=CountVectorizer()


from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier


svmc = SVC(kernel='sigmoid', gamma=1.0)
knn = KNeighborsClassifier(n_neighbors=49)
mnb = MultinomialNB(alpha=0.2)
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
lrgc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=31, random_state=111)
adb = AdaBoostClassifier(n_estimators=62, random_state=111)
bgc = BaggingClassifier(n_estimators=9, random_state=111)
etc = ExtraTreesClassifier(n_estimators=9, random_state=111)


clfs = {'SVC' : svmc,'KN' : knn, 'NB': mnb, 'DT': dtc, 'LR': lrgc, 'RF': rfc, 'AdaBoost': adb, 'BgC': bgc, 'ETC': etc}


X_train, X_test, y_train, y_test = train_test_split(df['sms'],df['labels'],random_state=1)
print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))


training_data=count_vector.fit_transform(X_train)
testing_data=count_vector.transform(X_test)


pred_scores = {}
for k,v in clfs.items():
    v.fit(training_data, y_train)
    predictions=v.predict(testing_data)
    pred_scores[k]=(( [accuracy_score(y_test,predictions), precision_score(y_test,predictions), recall_score(y_test,predictions), f1_score(y_test,predictions)]))
    

df = pd.DataFrame.from_dict(pred_scores,orient='index', columns=['Accuracy','Precision','Recall','F1'])

print(df)


df.plot(kind='bar', ylim=(0.8,1.0), figsize=(11,6), align='center', colormap="Accent")
plt.xticks(np.arange(9), df.index)
plt.ylabel('Score')
plt.title('Distribution by Classifier')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



print("\n\n[+] lets test with other unique msgs other than the datasets used: ") 
t=[input("[+]enter a text msg to test : ")]
t=np.array(t)
t=count_vector.transform(t)
prediction = mnb.predict(t)
if prediction == 1:
    print("HAAM!")
else:
    ##print("\033[1;31;40m [+]SPAAAM!")
    print("SPAAM!")


    





