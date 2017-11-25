from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import os
import numpy as np
from keras.models import load_model
from keras import backend
from scipy.ndimage import imread
from scipy.misc import imresize

@csrf_exempt
def validate(request):
	files = request.FILES
	myfile = request.FILES['image']
	fs = FileSystemStorage()
	path = "buffer/" + myfile.name
	filename = fs.save(path, myfile)
	resolution = (200, 200)
	img = imread(fname=path, flatten=True)
	resized = imresize(img, resolution)
	flatten = np.array([resized.flatten()])
	input_features = np.expand_dims(flatten, axis=2)
	model = load_model('model/saved_model.h5')
	res = model.predict(input_features, verbose=1)
	prob = res[0][0]
	backend.clear_session()
	os.remove(path)
	response = {"probability": str(prob)}
	return JsonResponse(response)