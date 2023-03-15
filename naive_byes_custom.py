# %%
import numpy as np
import pandas as pd
from nltk.corpus import brown

# %%
brown.categories()

# %%
words = brown.words()
sz = np.array(words).size
sz

# %%
sent = brown.sents(categories="romance")
" ".join(sent[7])

# %%
corpus = ['To them he could have been the broken bell in the church tower which rang before and after Mass , and at noon , and at six each evening -- its tone , repetitive , monotonous , never breaking the boredom of the streets .',
        'To them he could have been the broken bell in the church tower which rang before and after Mass , and at noon , and at six each evening -- its tone , repetitive , monotonous , never breaking the boredom of the streets .',
        'Yet if he were not there , they would have missed him , as they would have missed the sounds of bees buzzing against the screen door in early June ; ;',
        'They never called him by name , although he had one .'
        ]

# %%
from nltk.tokenize import sent_tokenize,word_tokenize

# %%
document = "my email id is manojkr2002@gmail.com"
tokenized = word_tokenize(document)
tokenized

# %% [markdown]
# TOKENIZATION .....

# %%
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('[a-zA-Z0-9@]+')
tokenizer.tokenize(document)

# %%
from nltk.corpus import stopwords


# %%

sent = word_tokenize(corpus[1])
sz = np.array(sent)
sz.size

# %%
sw = set(stopwords.words("english"))
def remove(docs,stopwords):
    words = [word for word in docs if word not in stopwords]
    return words

# %%
words = remove(sent,sw)
sz1 = np.array(words).size
sz1

# %% [markdown]
# STEMMING....

# %%
from nltk.stem import PorterStemmer

# %%
ps = PorterStemmer()
def stem(docs,port_stem):
    words = []
    for word in docs:
        w=ps.stem(word)
        words.append(w)
    return words
#EXAMPLE OF HOW STEMMER WORKS ALL ABOVE ARE BASICALLY A TECHNIQUE OF DATA CLEANING

# %%
words = stem(sent,ps)
words

# %% [markdown]
# COUSTOM IMPLIMENTATION OF NAIVE BYES

# %%
data = pd.read_csv("../data_used/mushrooms.csv")
data.head()

# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# %%
le = LabelEncoder()
data_m = data.apply(func = le.fit_transform)
data_m.head()
#LABELENCODER ENCODE/CHANGE ALL IN DIFFERENT NUMERIC CLASSES

# %%
data_m = np.array(data_m)
y = data_m[:,0]
X = data_m[:,1:]

# %%
X.shape , y.shape

# %%
X_train , X_test , y_train , y_test = train_test_split(X,y,train_size=0.33,random_state=42)

# %%
X_test.shape , y_test.shape

# %%
class naive:
    def fit(self,X,y):
        self.X_train = X 
        self.y_train = y
    def prior_probab(self,label):
        nume = np.sum(self.y_train == label)
        deno = self.y_train.shape[0]
        return nume/float(deno)
    def conditionl_probab(self,n_faeture,feature_value,label):
            X_modi = self.X_train[self.y_train == label]
            nume = np.sum(X_modi[:,n_faeture]==feature_value)
            return nume/float(X_modi.shape[0])
    def predict_point(self,X):
        self.X_test = X
        # self.y_test = y
        labels = np.unique(self.y_train)
        for label in labels:
            ll = 1.0
            post_probab = []
            for col in range(self.X_train.shape[1]):
                cp = self.conditionl_probab(col,X_test[col],label)
                ll = cp*ll
            prior_probab = self.prior_probab(label)
            post = prior_probab*ll
            post_probab.append(post)
        return np.argmax(post_probab)
    def predict(self,X):
        self.X_test = X
        ans = []
        for x in self.X_test:
            ans.append(self.predict_point(x))
        return np.array(ans)
    def score(self,X_test,y_test):
        self.X_test = X
        self.y_test = y
        return (self.predict(X_test)==y_test).mean()


# %%
model = naive()
model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_test,y_test)


