import pyodbc as po
import pandas as pd
#import sqlite3
#import mysql.connector
#import mysql.connector
#from mysql.connector import Error
#from mysql.connector import Error
#import re, nltk
import pyodbc as po
import pandas as pd
from datetime import datetime, timedelta

#import sqlite3
#import mysql.connector
#from mysql.connector import Error
#import re, nltk
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
# Machine Learning
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import LinearSVC
#from sklearn.model_selection import cross_val_predict
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_score, recall_score
#from sklearn.metrics import f1_score
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import classification_report
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
english_stops = stopwords.words('english')
import numpy as np
# Connection variables

    # Connection string
cnxn = po.connect('Driver={SQL Server Native Client 11.0};SERVER=' +
server+';DATABASE='+database+';UID='+username+';PWD=' + password)
cursor = cnxn.cursor()


storedProc = " EXEC EMS.SpGetAccountIdListIdTimeDifferenceForAnalysis"
result=pd.read_sql_query(storedProc,cnxn)
#cursor.execute(storedProc)
df2=pd.DataFrame(result,columns=['ACCOUNTID','LISTID'])
print(df2)

#median
def my_median(sample):
    n = len(sample)
    index = n // 2
     # Sample with an odd number of observations
    if n % 2:
         return sorted(sample)[index]
     # Sample with an even number of observations
    return sum(sorted(sample)[index - 1:index + 1]) / 2



for index,row in df2.iterrows():
    storedProc1 = "EXEC EMS.SpGetAggregateCampaignDataForAnalysis @AccountId="+str(row['ACCOUNTID'])+",@ListId="+str(row['LISTID'])+""
    result1=pd.read_sql_query(storedProc1,cnxn)
    #cursor.execute(storedProc1)
    df1=pd.DataFrame(result1,columns=['ListID',	'SegmentID',	'CampaignName','AccountID',	'CampaignSubject'	,'List'	,'Segment','Sent'	,'Delivered',	'Open',	'Click',	'Bounced',	'UniqueOpen',	'UniqueClick',	'Unsubscribe',	'Spam'	,'VirtualSpam',	'SendDate'])
    print(df1)
    data=df1
    data["Percentage_open"]=(pd.to_numeric(data['UniqueOpen']/data['Delivered']))*100
    data=data.sort_values("Percentage_open",axis=0,ascending=False)
    value_median=my_median(data["Percentage_open"])
#print(value_median)
    final_median=(20/100)*value_median
#final_median
    sd_median=value_median-final_median
    f=np.percentile(data["Percentage_open"], sd_median)
    data["Open_Percentage"]=np.where(data['Percentage_open']>=f,'1','0')

    from sklearn.feature_extraction.text import TfidfTransformer
    cv=TfidfVectorizer(analyzer='word',min_df=.005, max_df=.9,ngram_range=(2,3))
    review_tf = cv.fit_transform(data['CampaignSubject'])
    review_tf_nd = review_tf.toarray()
    df = pd.DataFrame(review_tf_nd, columns=cv.get_feature_names())
    print(df)
    model = pd.merge(data, df, left_index=True, right_index=True)
#file=model.to_csv('D:/new/result.csv')
    ml_model=model.drop(['ListID',	'SegmentID',	'CampaignName','AccountID',	'CampaignSubject'	,'List'	,'Segment','Sent'	,'Delivered',	'Open',	'Click',	'Bounced',	'UniqueOpen',	'UniqueClick',	'Unsubscribe',	'Spam'	,'VirtualSpam',	'SendDate','Percentage_open'],axis=1)
    print('hi')
#for i in model.columns:
    #print([i])
#ml_model=model.drop(["CampaignName"])
# Create X & y variables for Machine Learning
    X = ml_model.drop('Open_Percentage', axis=1)
    y = ml_model['Open_Percentage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
    def model(mod, model_name, x_train, y_train, x_test, y_test):
        mod.fit(x_train, y_train)
        print(model_name)
        
        #acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 5)
        #predictions = cross_val_predict(mod, X_train, y_train, cv = 5)
        #print("Accuracy:", round(acc.mean(),3))
        #cm = confusion_matrix(predictions, y_train)
        #print("Confusion Matrix:  \n", cm)
        #print("                    Classification Report \n",classification_report(predictions, y_train))

# 2. Random Forest Classifier
    ran = RandomForestClassifier(random_state=0)
    model(ran, "Random Forest Classifier", X_train, y_train, X_test, y_test)
# Compile arrays of columns (words) and feature importances
    fi = {'Words':ml_model.drop('Open_Percentage',axis=1).columns.tolist(),'Importance':ran.feature_importances_}


# Bung these into a dataframe, rank highest to lowest then slice top 20
    Importances = pd.DataFrame(fi).sort_values('Importance',ascending=False).head(20)
    data['wordsCount'] = data['CampaignSubject'].apply(lambda x: len(str(x).split(" ")))
    data['Date']=pd.to_datetime(data['SendDate'])
    data["SendDay"] = pd.to_datetime(data.Date, format="%Y/%m/%d").dt.day_name()
    data["SendMonth"] = pd.to_datetime(data.Date, format="%Y/%m/%d").dt.month
    data['hour'] = data['Date'].dt.hour
    group_words=data.groupby('wordsCount',as_index=False)["Percentage_open"].count()
        
    group_words=pd.DataFrame(group_words).sort_values(by=['Percentage_open'],ascending=False).head(10)
    group_day=data.groupby('SendDay',as_index=False)["Percentage_open"].count()
    group_day=pd.DataFrame(group_day).head(10)
    group_hour=data.groupby('hour',as_index=False)["Percentage_open"].count()
        
    group_hour=pd.DataFrame(group_hour).head(10)

    
    

    h=group_hour['hour'].to_list()
    h = str(h)[1:-1]
    h="".join(h.split())
    h="'{}'".format(h)

    w=group_words['wordsCount'].to_list()
    w=str(w)[1:-1]
    w="".join(w.split())
    w="'{}'".format(w)

    d=",".join(group_day['SendDay'])
    d="'{}'".format(d)
    words=','.join(Importances['Words'])
    words="'{}'".format(words)
    #store in sp
    hh=group_hour.values.tolist()
    hh=str(hh)
    import re
    import json
    dd=group_day[['SendDay','Percentage_open']]
    dd=dd.values.tolist()
    dd=json.dumps(dd)
    dd=re.sub(r'(?<!: )"(\S*?)"', '\\1', dd)
    dd="'{}'".format(dd)
    ff=group_hour.values.tolist()
    ff=str(ff)
    ff="'{}'".format(ff)
    
    
    ww=group_words.values.tolist()
    ww=str(ww)
    Importances=Importances['Words'].astype(str)
    
    sql="EXEC EMS.SpSaveCampaignAggregateAnalysisReport @AccountID="+str(row['ACCOUNTID'])+",@ListId="+str(row['LISTID'])+",@Days="+dd+",@Words="+words+",@Hours="+ff+",@WordCount="+w+""
    #cursor.execute(sql).commit()
    print(sql)
    
    
    
    
    
    
    
    

    #sql="EXEC EMS.SpSaveCampaignAggregateAnalysisReport @AccountID=?,@ListId=?,@Days=?,@Words=?,@Hours=?,WordCount=?"
    #params=(str(row['ACCOUNTID']),str(row['LISTID']),str(group_day),str(group_words),str(group_hour),str(Importances))
    
    
    



    