
from flask import Flask, request, render_template
import numpy as np
from sklearn.linear_model import LogisticRegression




from sklearn.feature_extraction.text import CountVectorizer
import joblib

app = Flask(__name__)
smpo=joblib.load(open('smpo.joblib','rb'))
count_vector=CountVectorizer()
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def predict():
    text_msg = request.form["sms"]
    text_msg = np.array(text_msg)
    my_msg = count_vector.transform(text_msg)
    prediction =  smpo.predict(my_msg)
    x="hii"
    if prediction[0] == 0:
        x = "Not a spam, it's ok "
    else:
        x = "it's a spam"
    return render_template('index.html', prediction=x)


if __name__ == "__main__":
    app.run(debug=True)  
