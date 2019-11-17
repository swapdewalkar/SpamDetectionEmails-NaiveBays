import os
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

#Common Functions:
def get_emails(path):
    documents = os.listdir("./EmailsData/"+path)
    emails=[]
    total=len(documents)
    for file in documents:
        f = open("./EmailsData/"+path+"/" + file)
        emails.append(f.read())
    return emails,total

def get_vectorizer(text):
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(text)
    return vectorizer

#Trainings
non_spam,total_non_spam=get_emails("nonspam-train")
non_spam_vectorizer=get_vectorizer(non_spam)
non_spam_vocab=non_spam_vectorizer.vocabulary_

spam,total_spam=get_emails("spam-train")
spam_vectorizer=get_vectorizer(spam)
spam_vocab=spam_vectorizer.vocabulary_

total_email=total_non_spam+total_spam
prob_spam=math.log10(float(total_spam)/total_email)
prob_non_spam=math.log10(float(total_non_spam)/total_email)


#Conditional Probabilty Function
def prob_of_words_given_spam(words):
    total=0
    for word in words:
        if word in spam_vocab:
            index=spam_vocab.keys().index(word)
            value=spam_vectorizer.idf_[index]
            total=total+math.log10(value)
    return total

def prob_of_words_given_non_spam(words):
    total=0
    for word in words:
        if word in non_spam_vocab:
            index=non_spam_vocab.keys().index(word)
            value=non_spam_vectorizer.idf_[index]
            total=total+math.log10(value)
    return total

#Testing
    #positive in spam= 1
    #negative in non-spam=0
TP=TN=FN=FP=0
test_non_spam,test_total_non_spam=get_emails("nonspam-test")
test_spam,test_total_spam=get_emails("spam-test")
test_non_spam=[(i,0) for i in test_non_spam]
test_spam=[(i,1) for i in test_spam]
all_test_email_label=test_non_spam+test_spam

i=0
for email,label in all_test_email_label:
        email_vectorizer=get_vectorizer(email.split(" "))
        words= email_vectorizer.vocabulary_
        spam_prob=prob_of_words_given_spam(words)
        non_spam_prob=prob_of_words_given_non_spam(words)
        if spam_prob>non_spam_prob:
            print "Email "+str(i)+": spam"
            if label==1:
                TP=TP+1
            else:
                FP=FP+1
        else:
            print "Email "+str(i)+": non spam"
            if label==0:
                TN=TN+1
            else:
                FN=FN+1
        i=i+1

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
# plt.plot([0, 1], [0, 1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()