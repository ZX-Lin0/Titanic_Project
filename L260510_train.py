import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

train_url = 'https://raw.githubusercontent.com/cystanford/Titanic_Data/master/train.csv'
test_url = 'https://raw.githubusercontent.com/cystanford/Titanic_Data/master/test.csv'
train = pd.read_csv(train_url)
test = pd.read_csv(test_url)
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Sex'] = train['Sex'].map({'male':0, 'female':1})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = train[features]
y = train['Survived']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
print('Model trained and saved.')