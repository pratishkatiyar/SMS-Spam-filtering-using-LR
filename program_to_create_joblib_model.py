import numpy as np 
import pandas as pd 
import joblib

from sklearn.model_selection import train_test_split


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score




filepath='spam.csv'
df=pd.read_csv(filepath, encoding='latin-1')


df=df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)
df=df.rename(columns={'v1':'labels','v2': 'sms'})



df['labels']=df.labels.map({'spam':0, 'ham':1})

    
    

count_vector=CountVectorizer()





from sklearn.linear_model import LogisticRegression


lrgc = LogisticRegression(solver='liblinear', penalty='l1')



X_train, X_test, y_train, y_test = train_test_split(df['sms'],df['labels'],random_state=1)
print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))



training_data=count_vector.fit_transform(X_train)
testing_data=count_vector.transform(X_test)



lrgc.fit(training_data, y_train)


joblib.dump(lrgc, open('smpo.joblib','wb'))


smpo=joblib.load(open('smpo.joblib','rb'))

predictions=lrgc.predict(testing_data)
lrgc_acc = accuracy_score(y_test,predictions)

print(lrgc_acc*100)



print("\n\n[+] lets test with other unique msgs other than the datasets used: ") 
t=[input("[+]enter a text msg to test : ")]
t=np.array(t)
t=count_vector.transform(t)
prediction = lrgc.predict(t)
if prediction == 1:
    print("HAAM!")
else:
    ##print("\033[1;31;40m [+]SPAAAM!")
    print("SPAAM!")


    





