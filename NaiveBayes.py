import os
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.naive_bayes import GaussianNB
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
#
# #Common Functions:
def get_emails(path):
    documents = os.listdir("./EmailsData/"+path)
    emails=[]
    total=len(documents)
    for file in documents:
        f = open("./EmailsData/"+path+"/" + file)
        emails.append(f.read())
    return emails

#non_spam==0
#spam==1
train_labels=numpy.append(np.zeros(350),np.ones(350))
non_spam=get_emails("nonspam-train")
spam=get_emails("spam-train")
train=non_spam+spam
vectorizer = TfidfVectorizer()
v=vectorizer.fit_transform(train)

ch2=SelectKBest(chi2,k=50)
s=ch2.fit_transform(v,train_labels)
model1 = GaussianNB()
model1.fit(s.toarray(),train_labels)

non_spam_t=get_emails("nonspam-test")
spam_t=get_emails("spam-test")
test=non_spam_t+spam_t
v=vectorizer.transform(test)
s=ch2.transform(v)
s=s.toarray()

TP=TN=FP=FN=0
for i in s[:130]:
    p=model1.predict([i])
    if p[0]==0:
        TN=TN+1
    else:
        FN=FN+1
for i in s[130:]:
    p=model1.predict([i])
    if p[0]==1:
        TP=TP+1
    else:
        FP=FP+1

print TP,FP,TN,FN
print "Confusion Matrix"
print str(TP)+"|"+str(FP)
print "--------"
print str(FN)+"|"+str(TN)
Accuracy=((TP+TN)/float(TP+TN+FP+FN))
Recall=float(TP)/(TP+FN)
Precision=float(TP)/(TP+FP)
F1_Score=2*((Precision*Recall)/float(Precision+Recall))
TPR=TP/float(TP+FN)
FPR=FP/float(FP+TN)
print "Accuracy: "+str(Accuracy)
print "Precision: "+str(Precision)
print "Recall: "+str(Recall)
print "F1 Score: "+str(F1_Score)

lw = 2
plt.plot(FPR, TPR)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()