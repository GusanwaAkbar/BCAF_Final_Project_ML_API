from django.apps import AppConfig

from tensorflow import keras
import tensorflow as tf
from keras import models 


class MlAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_app'


    def ready(self):
        # Load your machine learning model when the application starts
        self.model = tf.keras.models.load_model('./model5') 
        self.model.load_weights('./model5')
