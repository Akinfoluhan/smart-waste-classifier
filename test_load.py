from keras.models import load_model

model = load_model("Resources/Model/keras_model.h5", compile=False)
model.save("Resources/Model/converted_model")
print("Model converted and saved successfully.")
