import os
import sys
import numpy as np
import pandas as pd
from Model import build_cf_model, rate



def predict_rating(trained_model, userid, movieid):
    return rate(trained_model, userid - 1, movieid - 1)


def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if len(directory) == 0: return
  if not os.path.exists(directory):
    os.makedirs(directory)


def main(argv):
    users = pd.read_csv(argv[3], sep='::', engine='python',
            usecols=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])

    movies = pd.read_csv(argv[2], sep='::', engine='python',
            usecols=['movieID', 'Title', 'Genres'])

    test_data = pd.read_csv(argv[0], usecols=['UserID', 'MovieID'])

    trained_model = build_cf_model(max_userid, max_movieid, DIM)
    trained_model.load_weights(MODEL_WEIGHTS_FILE)

    recommendations = pd.read_csv(argv[0], usecols=['TestDataID'])
    recommendations['Rating'] = test_data.apply(lambda x: predict_rating(trained_model, x['UserID'], x['MovieID']), axis=1)
    

    ensure_dir(argv[1])
    recommendations.to_csv(argv[1], index=False, columns=['TestDataID', 'Rating'])
    print("Done")

if __name__ == '__main__':
    

    MODEL_DIR = './model'
    MODEL_WEIGHTS_FILE = 'weights.h5'

    DATA_DIR = "./"

    MODEL_WEIGHTS_FILE = os.path.join(MODEL_DIR, MODEL_WEIGHTS_FILE)
    MAX_CSV = os.path.join(MODEL_DIR, "max.csv")
    info = pd.read_csv(MAX_CSV)
    DIM = list(info['dim'])[0]
    max_userid = list(info['max_userid'])[0]
    max_movieid = list(info['max_movieid'])[0]

    sys.exit(main(sys.argv[1:]))
