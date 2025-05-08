import argparse
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, AllChem, Descriptors
from rdkit import DataStructs
from rdkit.Chem import Descriptors

import chemprop
from chemprop.args import TrainArgs, PredictArgs
from chemprop.train import cross_validate, run_training, make_predictions

from chemproppred.make_screen_data import make_screening_data
from chemproppred.screen_polys import screen_poly


PATH_CHEM = os.getcwd()
DATADIR = f"{PATH_CHEM}/data/cross_val_data"
# TYPE = "arr"
MODELDIR = f"{PATH_CHEM}/models/screen"
SAVEPATH = f"{PATH_CHEM}/data/polyinfo_3"
PREDS_PATH = f"{DATADIR}/preds_screen_1"
#! len(_screen)= 10433, len(_full)= 10015. 

def train_and_predict(data_path, model_path, preds_path, gpu='false', gpu_number=0):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    TRAIN = f"{data_path}/s_full.csv"           ## also trained on the clean_train_data. 
    TRAINFEATS = f"{data_path}/f_full.csv"      

    TEST = f"{data_path}/s_screen.csv"      ## also comes from the clean_train_data.  
    TESTFEATS = f"{data_path}/f_screen.csv"

    PREDS=  f"{preds_path}/preds_screen.csv"        ## full-lipo chain at the beginning. the reason why results is wrong. 
    SAVEDIR = model_path

    #train chemprop model
    argument = [
    "--data_path", f"{TRAIN}",
    "--features_path", f"{TRAINFEATS}",
    "--save_dir", f"{SAVEDIR}",
    "--dataset_type", "regression",         ## still using regression model
    "--split_size", "0.9", "0.05", "0.05", ## use a different split size.
    "--metric", "mae",
    "--extra_metrics", "r2", "rmse", "mse",  
    "--num_folds", "1",
    # "--arr", 
    # "--arr_vtf", "arr", 
    # "--target_columns", "ECW", "temperature",  ##// has been proved to be uesless. it points everything other than smiles in s_full. 
    # "--ignore_columns", "conductivity",         #! when the --arr is not used, temperature is working as targets. (arr, temp is ignored)
    # "--task_names", ["conductivity","ECW"],   # // useless. and no this flag actually.
    # "--num_tasks", "2",        
    "--depth", "2",
    "--dropout", "0",
    "--ffn_num_layers", "3",
    "--hidden_size", "2400",
    "--epochs", "25",       ## set epoches from 25 to 5
    "--pytorch_seed","5"
    # "--predict_ecw", 
    # "--ecw_loss_weight", "0.3",
    # "--ecw_column", "ECW"
    ]
    
    if gpu=='true':
        argument.append("--gpu")
        argument.append(str(gpu_number))
    else:
        argument.append("--no_cuda")

    train_args = TrainArgs().parse_args(argument)

    cross_validate(args=train_args, train_func=run_training) ## still using the ChemProp architecture !!!

    pred_args = [                         ## this is the prediction part, but now I just want to train the model.          
        "--test_path", f"{TEST}",
        # "--dataset_type", "regression",
        "--features_path", f"{TESTFEATS}",
        "--checkpoint_dir", f"{SAVEDIR}",
        "--arr_vtf", "arr",
        "--preds_path", f"{PREDS}", ## generated from the test file (screen_prediction.csv) 
    ]

    make_predictions(args=PredictArgs().parse_args(pred_args)) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing input parameters for cross validation training')
    parser.add_argument('--make_data', choices=['true', 'false'], default='false', 
                        help='Determines whether the data should be generated or not')
    parser.add_argument('--train_predict', choices=['true', 'false'], default='false',
                        help='Train the model on all the data and predict on polyinfo data')
    parser.add_argument('--polyinfo_datafiles', choices=['true', 'false'], default='false',
                        help='Generate easily viewable files for the polyinfo data from the predicitons')
    parser.add_argument('--gpu', choices=['true', 'false'], default='false',
                        help='The model is trained on cuda enabled GPU, default false - training on CPU')
    parser.add_argument('--gpu_number', default='0',
                        help='Cuda device, if no GPU leave this empty')
    args = parser.parse_args()
    
    if args.make_data == "true":
        print("Creating the cross validation data files for training!")
        make_screening_data(DATADIR, data_path=f'{PATH_CHEM}/data/clean_train_data_ECW.csv', name="screen") ##can't find f_screen, rename it. 
    if args.train_predict == "true":
        print("Training loop begins!")
        train_and_predict(DATADIR, MODELDIR, PREDS_PATH, args.gpu) 
    if args.polyinfo_datafiles == "true":
        screen_poly(DATADIR, SAVEPATH, PREDS_PATH)      ## from the PREDS_PATH output, generate a better output file(html)
        


        ## python train_pred_ECW.py --make_data true --train_predict true --polyinfo_datafiles true --gpu true --gpu_number 1
        #! in this case, only train on the ECW data. and do not use the conductivity data.