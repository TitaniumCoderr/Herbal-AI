from tensorflow.keras.models import load_model, save_model

# Load the HDF5 model
model = load_model("saved_model/myModel.hdf5")

# Save it in the SavedModel format
save_model(model, "saved_model/")

