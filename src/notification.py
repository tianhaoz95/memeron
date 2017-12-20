import firebase_admin
from firebase_admin import credentials
from .config import *

cred = credentials.Certificate(key_path)
firebase_admin.initialize_app(cred)
