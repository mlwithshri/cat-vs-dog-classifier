from tensorflow import keras
import tensorflow as tf
import numpy as np


model = keras.models.load_model("cats_vs_dogs_model2 .keras")

class_names = ["cats", "dogs"]


# 3. Function to preprocess and predict
def predict_image(img_path, img_size=(128, 128)):
    # Load and preprocess image
    img = keras.utils.load_img(img_path, target_size=img_size)
    img_array = keras.utils.img_to_array(img) / 255.0   # normalize
    img_array = np.expand_dims(img_array, axis=0)       # add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    print(f"Image: {img_path}")
    print(f"Prediction → {predicted_class.upper()} (confidence: {confidence:.2f})")


# 4. Run prediction on sample image
if __name__ == "__main__":
    img_path = "cat2.jpg"   # put your image file path here
    predict_image(img_path)
