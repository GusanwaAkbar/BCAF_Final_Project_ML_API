from django.shortcuts import render
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .preprocessing import preprocess

# from tensorflow import keras
import tensorflow as tf
# from keras import models 


import numpy as np

import json

from sklearn.preprocessing import LabelEncoder
import joblib
import pickle

from django.apps import apps

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt  # Disable CSRF protection for this view (for demonstration purposes)
def predict_csv(request):

    if request.method == "POST":
        data = json.loads(request.body.decode('utf-8'))

        df = pd.read_json(json.dumps(data))
        #df = pd.read_csv('./dummy2.csv')
        print(df)

        ml_app = apps.get_app_config('ml_app')
        model = ml_app.model


        # Example datasetpyth_texts
        dataset_texts = ["SHF", "Settlement", "Fidusia", "Pinalty", "UMK"]

        
        # Preprocess the data
        df = preprocess.preprocess(df, dataset_texts)
        # model = tf.keras.models.load_model('./model5') 
        # model.load_weights('./model5')

        print("-------------")

        df = preprocess.preprocess(df, dataset_texts)
        df

        print(df)

        numeric_feature_names = ['closest_words_num', 'nominal']
        numeric_features = df[numeric_feature_names]
        numeric_features = tf.convert_to_tensor(numeric_features)

        predict = model.predict(numeric_features)
        print("==============")
        print(predict)

        # Load the label encoder object from the file
        label_encoder = joblib.load('label_encoder.pkl')


        predicted_labels = []
        for prediction in predict:
            predicted_index = np.argmax(prediction)  # Find the index of the class with the highest probability
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]  # Map the index to the original class label
            predicted_labels.append(predicted_label)

        print("Predicted labels:", predicted_labels)


        
        predicted_labels_list = predicted_labels

        # Prepare response data
        response_data = {
            'predicted_labels': predicted_labels_list
        }

        # Return JSON response
        return JsonResponse(response_data)



    return JsonResponse({'success': True, 'message': 'JSON data processed successfully'})



@csrf_exempt  # Disable CSRF protection for this view (for demonstration purposes)
def predict_single_data(request):

    if request.method == "POST":
        data = json.loads(request.body.decode('utf-8'))
        print(data)

        print("data succec------=====================================")

        ml_app = apps.get_app_config('ml_app')
        model = ml_app.model

        dataset_texts = ["SHF", "Settlement", "Fidusia", "Pinalty", "UMK"]

        label_encoder = joblib.load('label_encoder.pkl')

        deskripsi = data.get('deskripsi', '')  # Assuming 'deskripsi' is the key for deskripsi in the JSON payload
        nominal = data.get('nominal', 0)  # Assuming 'nominal' is the key for nominal in the JSON payload


        # Preprocess input
        input_data = preprocess.preprocess_input(deskripsi, nominal, dataset_texts)

        # Predict using the model
        predictions = model.predict(input_data)

        # Decode the predicted labels
        predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])

        print("Predicted label:", predicted_label)

        label = predicted_label.tolist()

        # Prepare response data
        response_data = {
            'predicted_labels': label
        }

        print("respon data adalah")
        print(response_data)

        # Return JSON response
        return JsonResponse(response_data)

    return JsonResponse({'success': True, 'message': 'JSON data processed successfully'})



