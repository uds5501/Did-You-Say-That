x=str(input('Please enter the whatsapp chat file :'))
text=open(x,'r')
aux=[line for line in text]
users=[]
def getAuxStr(arr):
    aux=""
    for i in arr:
        aux=aux+" "+str(i)
    return aux.strip()

def findUser(arr):
    for i in range(2,len(arr)):
        if arr[i][-1]==":":
            aux_string=getAuxStr(arr[3:i])
            users.append((aux_string+" "+arr[i].split(':')[0]).strip())
for i in aux:
    findUser(i.split())

threshold=10
verified=[]
for i in set(users):
    count=0
    for j in users:
        if i==j:
            count+=1
    if count>threshold:
        verified.append(i)

#declare lists
messeges=[]
sender=[]
date=[]
#note : 0,1 are date and time. 2:x are name and x: is messege

def splitter(mess_array):
    #leave time
    for i in range(2,len(mess_array)-1):
        if mess_array[i][-1]==":":
            aux_string=getAuxStr(mess_array[3:i])
            final_user_string=(aux_string+" "+mess_array[i].split(':')[0]).strip()
            if final_user_string in verified:
                sender.append(final_user_string)
                date.append(mess_array[0])

                aux_str2=""
                for j in range(i+1,len(mess_array)):
                    aux_str2=aux_str2+" "+str(mess_array[j])
                messeges.append(aux_str2.strip())

for i in range(1,len(aux)):
    splitter(aux[i].split())
import pandas as pd
import numpy as np

mess_series=pd.Series(v for v in messeges)
sender_series=pd.Series(i for i in sender)
date_series=pd.Series(k for k in date)
df_raw=pd.concat([sender_series,mess_series,date_series],axis=1)
df_raw['Length']=df_raw['Messeges'].apply(len)
names=pd.get_dummies(df_raw['Sender'],drop_first=True)
df_final=pd.concat([df_raw,names],axis=1)
def giveName():
    main=['Messeges','Length','Month']
    for i in df_final.columns:
        if i not in main:
            return i



from nltk.corpus import stopwords
import string

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

from sklearn.model_selection import train_test_split
to_be_used=giveName()
msg_train, msg_test, label_train, label_test =train_test_split(df_final['Messeges'],df_final[to_be_used], test_size=0.2)

pipeline.fit(msg_train,label_train)
from sklearn.metrics import classification_report
pred=pipeline.predict(msg_test)

print (classification_report(pred,label_test))

