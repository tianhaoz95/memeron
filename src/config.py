# utility parameters
mode = 'overnight'
# mode = 'quick'
add_samples_dir = 'addsample/samples.txt'
tmp_dir = 'tmp'
dataset_dir = 'dataset'
model_dir = 'model'
saved_model_filename = 'saved_model.h5'
saved_weights_filename = 'js_model_weights.hdf5'
saved_structure_filename = 'js_model_structure.json'
meta_dir = 'src/source.json'
backup_dir = 'src/backup.json'
plot_dir = 'plots'
label_encoder = {'meme': [1, 0], 'nonmeme': [0, 1]}
width = 100
height = 100
resolution = (width, height)
key_path = 'key.json'
username = 'tianhaoz@umich.edu'

# neural network parameters
conv_input_shape = (width, height, 3)
train_batch_size = 10
test_batch_size = 10
epochs = 1
iter_cnt = 100
checkpoint = 5
sample_size = 50

# grid search parameters
conv_layer_cnts = [2, 3, 4, 5, 6, 7]
fc_layer_cnts = [1, 2, 3]
neuron_cnts = [32, 64, 128, 256, 512]

custom_combinations = [
{'conv': [32, 64, 128], 'fc': [64, 64]},
{'conv': [32, 32, 128], 'fc': [128, 64]}
]

# testing configuration

# clean_up_dataset = True
clean_up_dataset = False
