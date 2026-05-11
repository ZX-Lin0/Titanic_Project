import joblib
import pandas as pd
model=joblib.load('model.pkl')
def predict_survival(pclass, sex, age, sibsp, parch, fare):
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]],columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
    return model.predict(input_data)[0]
if __name__=='__main__':
    result=predict_survival(3,1,25,0,0,7.25)
    print(f'Survived:{result}')