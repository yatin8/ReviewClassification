from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB

tokenizer=RegexpTokenizer(r'\w+')
en_stopwords=set(stopwords.words('english'))
ps=PorterStemmer()

def getCleanReview(review):
     review=review.lower()
     review=review.replace("<br /><br />"," ")

     tokens=tokenizer.tokenize(review)
     newTokens=[t for t in tokens if t not in en_stopwords]
     stemmed_tokens=[ps.stem(t) for t in newTokens]
     cleanedReview=' '.join(stemmed_tokens)
     return cleanedReview

x = ["This was awesome an awesome movie",
     "Great movie! I liked it a lot",
     "Happy Ending! awesome acting by the hero",
     "loved it! truly great",
     "bad not upto the mark",
     "could have been better",
     "Surely a Disappointing movie"]

y = [1,1,1,1,0,0,0] # 1 - Positive, 0 - Negative Class

x_test=["I was happy & happy and I loved the acting in the movie",
          "The movie I saw was bad"]

x_clean = [getCleanReview(r) for r in x]
x_test_clean = [getCleanReview(r) for r in x_test]
cv=CountVectorizer(ngram_range=(1,1))
x_vec=cv.fit_transform(x_clean).toarray()
x_test_vec=cv.transform(x_test_clean).toarray()

#MULTINOMIAL
mnb=MultinomialNB()
mnb.fit(x_vec,y)
print(mnb.predict(x_test_vec))
print(mnb.predict_proba(x_test_vec))
print(mnb.score(x_vec,y))


#BERNOULLI
bnb=BernoulliNB()
bnb.fit(x_vec,y)
print(bnb.predict(x_test_vec))
print(bnb.predict_proba(x_test_vec))
print(bnb.score(x_vec,y))

#GAUSSIAN
gnb = GaussianNB()
gnb.fit(x_vec,y)
print(gnb.predict(x_test_vec))
print(gnb.predict_proba(x_test_vec))
print(gnb.score(x_vec,y))
