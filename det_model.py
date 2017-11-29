from keras.models import Model, load_model
import argparse
from keras.utils import plot_model
ar = argparse.ArgumentParser()
ar.add_argument('path', type = str, help = "Please enter path to model")
args = ar.parse_args()
MODEL_NAME = args.path

model = load_model(MODEL_NAME+'/model.hdf5',compile=False)
print(model.summary())
plot_model(model, to_file=MODEL_NAME+'/model.png',show_shapes=True,show_layer_names=True)
    
