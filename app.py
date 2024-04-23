from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,  MinMaxScaler

app = Flask(__name__)

# Load the pre-trained model
model = load_model("fad_model1.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    profile_pic = float(request.form['profilepic'])
    nums_length = float(request.form['nums/length'])
    fullnamewords = float(request.form['fullnamewords'])
    nums_length_fullname = float(request.form['nums/lengthfullname'])
    name_username = float(request.form['nameusername'])
    description_length = float(request.form['descriptionlength'])
    external_url = float(request.form['externalurl'])
    private = float(request.form['private'])
    posts = float(request.form['posts'])
    followers = float(request.form['followers'])
    follows = float(request.form['follows'])

    # Preprocess the input data
    input_data = pd.DataFrame({
        'profile pic': [profile_pic],
        'nums/length username': [nums_length],
        'fullname words': [fullnamewords],
        'nums/length fullname': [nums_length_fullname],
        'name==username': [name_username],
        'description length': [description_length],
        'external URL': [external_url],
        'private': [private],
        '#posts': [posts],
        '#followers': [followers],
        '#follows': [follows]
    })

    # Normalize the input data
    instagram_df_train = pd.read_csv('train.csv')

    x_train = instagram_df_train.drop(columns = ['fake'])

    #Scaling the data before training the model (Normalize the data)
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(x_train)
    scaled_input_data = scaler_x.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input_data)
    predicted_class = np.argmax(prediction)


    # Return the prediction
    return "Fake" if predicted_class == 1 else "Real"

if __name__ == '__main__':
    app.run(debug=True)
