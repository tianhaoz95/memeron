import os
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

def load_img(dir_path, resolution, label):
	file_list = os.listdir(dir_path)
	res = []
	for i in range(len(file_list)):
		print("reading file ", str(i + 1), " out of ", len(file_list), " files ...")
		if file_list[i].endswith(".jpg") or file_list[i].endswith(".jpeg") or file_list[i].endswith(".png"):
			path = dir_path + "/" + file_list[i]
			img = imread(fname=path, flatten=True)
			resized = imresize(img, resolution)
			flatten = resized.flatten()
			row = np.append(flatten, np.array([label]))
			res.append(row)
		else:
			print("warning: not a image file")
	output = np.array(res)
	return output

def combine_datasets(datasets):
	combined = datasets[0]
	for i in range(1, len(datasets), 1):
		dataset = datasets[i]
		combined = np.vstack((combined, dataset))
	np.random.shuffle(combined)
	return combined

def test_error(y, pred, threshold):
	size = y.shape[0]
	cnt = 0
	for i in range(size):
		print("correct: ", y[i, 0], ", prediction: ", pred[i, 0])
		if y[i, 0] == 1:
			if pred[i, 0] >= threshold:
				print("matching prediction")
				cnt = cnt + 1
		if y[i, 0] == 0:
			if pred[i, 0] < threshold:
				print("matching prediction")
				cnt = cnt + 1
	print(cnt, " out of ", size, " prediction are correct")
	return cnt / size

def main():
	print("running main ...")
	resolution = (200, 200)
	positive_traning_set = load_img("data/training_positive", resolution, 1)
	negative_training_set = load_img("data/training_negative", resolution, 0)
	positive_testing_set = load_img("data/testing_positive", resolution, 1)
	negative_testing_set = load_img("data/testing_negative", resolution, 0)
	print("training positive: ", positive_traning_set.shape)
	print("training negative: ", negative_training_set.shape)
	print("testing positive: ", positive_testing_set.shape)
	print("testing negative: ", negative_testing_set.shape)
	combined_set = combine_datasets([positive_traning_set, negative_training_set])
	combined_test = combine_datasets([positive_testing_set, negative_testing_set])
	training_feature = np.expand_dims(combined_set[:,:-1], axis=2)
	training_label = combined_set[:,-1:]
	testing_feature = np.expand_dims(combined_test[:,:-1], axis=2)
	testing_label = combined_test[:,-1:]
	print("training feature shape: ", training_feature.shape)
	print("training label shape: ", training_label.shape)
	print("testing feature shape: ", testing_feature.shape)
	print("testing label shape: ", testing_label.shape)
	cur_list = os.listdir("model")
	model = None
	if "saved_model.h5" in cur_list:
		print("found previous model, recovering ...")
		model = load_model('model/saved_model.h5')
	else:
		print("no previous model exist, constructing new model ...")
		model = Sequential()
		model.add(Conv1D(100, 3, activation='relu', input_shape=(training_feature.shape[1], training_feature.shape[2])))
		model.add(Conv1D(100, 3, activation='relu'))
		model.add(MaxPooling1D(3))
		model.add(Conv1D(200, 3, activation='relu'))
		model.add(Conv1D(50, 3, activation='relu'))
		model.add(GlobalAveragePooling1D())
		model.add(Dropout(0.5))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	model.fit(training_feature, training_label, batch_size=16, epochs=10)
	model.save('model/saved_model.h5')
	model.save_weights('model/js_model_weights.hdf5')
	with open('model/js_model_structure.json', 'w') as f:
		f.write(model.to_json())
	training_score = model.evaluate(training_feature, training_label, batch_size=16)
	print("training error loss: ", training_score[0], ", training accuracy: ", training_score[1])
	testing_result = model.predict(testing_feature, verbose=0)
	print("testing result shape: ", testing_result.shape)
	print(testing_result[:10,:])
	print("correct result shape: ", testing_label.shape)
	print(testing_label[:10,:])
	testing_accuracy = test_error(testing_label, testing_result, 0.5)
	print("testing accuracy: ", testing_accuracy)

if __name__ == "__main__":
	main()