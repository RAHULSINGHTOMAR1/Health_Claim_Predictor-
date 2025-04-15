from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open("model.pkl",'rb') as f:
    model = pickle.load(f)

# Load the encoder
with open("encoder.pkl",'rb') as f:
    encoder_dict = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])
def predict():
    try:
        # get use input from 
        age = int(request.form["age"])
        gender = request.form['gender']
        state = request.form['state']
        claim_amount = float(request.form['claim_amount'])

        # Create dataframe from input
        input_data = pd.DataFrame([[age,gender, state, claim_amount]],
                                  columns = ["Patient_Age", "Gender",'State','Claim_Amount'])
        
        # Encode categorical feature
        input_data["Gender"] = encoder_dict['Gender'].transform(input_data["Gender"])
        input_data["State"] = encoder_dict['State'].transform(input_data["State"])

        # predict
        prediction = model.predict(input_data)[0]

        # convert prediction to readable message
        result_msg = "Claim Approved" if prediction==1  else "Claim Denied"

        return render_template("index.html",prediction=result_msg)
    
    except Exception as e:
        return jsonify({"error":str(e)})
    
if __name__ == "__main__":
    app.run(debug = True)
