import os
import sys
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from Model import build_cf_model




def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1., 5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))

def main(argv):
    ratings = pd.read_csv(argv[0],
                          usecols=['UserID', 'MovieID', 'Rating'])
    max_userid = ratings['UserID'].drop_duplicates().max()
    max_movieid = ratings['MovieID'].drop_duplicates().max()
    ratings['User_emb_id'] = ratings['UserID'] - 1
    ratings['Movie_emb_id'] = ratings['MovieID'] - 1

    maximum = {}
    maximum['max_userid'] = [max_userid]
    maximum['max_movieid'] = [max_movieid]
    maximum['dim'] = [DIM]
    pd.DataFrame(data=maximum).to_csv(MAX_FILE, index=False)

    ratings = ratings.sample(frac=1)
    Users = ratings['User_emb_id'].values
    Movies = ratings['Movie_emb_id'].values
    Ratings = ratings['Rating'].values

    model = build_cf_model(max_userid, max_movieid, DIM)
    model.compile(loss='mse', optimizer='adamax', metrics=[rmse])

    callbacks = [EarlyStopping('val_rmse', patience=2),
                 ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
    history = model.fit([Users, Movies], Ratings, epochs=1000, batch_size=256, validation_split=.1, verbose=1, callbacks=callbacks)


if __name__ == '__main__':
    

    MODEL_DIR = './model'
    MODEL_WEIGHTS_FILE = 'weights-000.h5'
    DATA_DIR = "./"
    DIM = 15
    MODEL_WEIGHTS_FILE = os.path.join(MODEL_DIR, MODEL_WEIGHTS_FILE)
    MAX_FILE = os.path.join(MODEL_DIR, "max.csv")

    sys.exit(main(sys.argv[1:]))
