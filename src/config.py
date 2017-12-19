# notification parameters
slack_api = 'xoxp-288942247782-288076142754-288116646020-dbcb3a6cf3ccb37631d992f2a515bbac'

# utility parameters
add_samples_dir = 'addsample/samples.txt'
tmp_dir = 'tmp'
dataset_dir = 'dataset'
model_dir = 'model'
saved_model_filename = 'saved_model.h5'
saved_weights_filename = 'js_model_weights.hdf5'
saved_structure_filename = 'js_model_structure.json'
meta_dir = 'src/source.json'
backup_dir = 'src/backup.json'
label_encoder = {'meme': [1], 'nonmeme': [0]}
width = 200
height = 200
resolution = (width, height)

# neural network parameters
conv_input_shape = (width, height, 3)
conv_kernel = (15, 15)
conv_pool_size = (3, 3)
conv_dropout = 0.25
fc_dropout = 0.25
output_size = 1
step_size = 0.05
decay = 1e-6
momentum = 0.9
train_batch_size = 1
test_batch_size = 1
epochs = 3
conv_layers = [64, 64]
fc_layers = [32]
iter_cnt = 30
sample_size = 50
bloss = 'binary_crossentropy'
closs = 'categorical_crossentropy'

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
