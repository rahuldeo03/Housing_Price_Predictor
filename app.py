from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('housing_price.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        rooms = int(request.form['rooms'])
        distance=float(request.form['distance'])
        prediction=model.predict([[rooms,distance]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this phone")
        else:
            return render_template('index.html',prediction_text="Predicted price is {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

