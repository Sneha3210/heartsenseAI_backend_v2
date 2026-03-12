import tensorflow as tf

print("Loading keras model...")
model = tf.keras.models.load_model("model/cnn_lstm_mitbih_final.keras", compile=False)

print("Exporting TensorFlow SavedModel...")
model.export("model/heartsense_model")

print("Saved successfully!")