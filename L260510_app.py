from flask import Flask,request,jsonify
import joblib
import pandas as pd
app=Flask(__name__)
model=joblib.load('model.pkl')
@app.route('/predict', methods=['POST'])
def predict():
    data=request.json
    input_df=pd.DataFrame([data['features']],columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
    pred=model.predict(input_df)[0]
    return jsonify({'survived':int(pred)})
if __name__=='__main__':
    app.run(debug=True)