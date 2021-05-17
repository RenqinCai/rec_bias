import pandas as pd
import gurobipy as gp
import os
from data import MOVIE 
from model import UGF
import argparse

def main(args):
    data_path = args.data_folder

    pred_file = data_path+"pred.pickle"

    group_file = data_path+"group_userid.json"

    data_obj = MOVIE(pred_file, group_file) 

    topk = args.topk
    epsilon = args.epsilon
    fairness_metric = args.fairness_metric
        
    model_obj = UGF(topk, epsilon, fairness_metric)
    model_obj.f_train(data_obj.m_eval_df, data_obj.m_group1_eval_df, data_obj.m_group2_eval_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### parameters of data
    
    parser.add_argument("--data_folder", type=str, default="../",
                        help="folder to store the data")
    parser.add_argument("--data_name", type=str, default="movie",
                        help="folder to store the data")

    ### parameters of model
    parser.add_argument("--model_name", type=str, default="bpr")


    ### fairness
    parser.add_argument("--fairness_metric", type=str, default="f1")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="fairness threshold")
    parser.add_argument("--group", type=str, default="activity", help="separate users into groups")

    parser.add_argument("--topk", type=int, default=20,
                        help="fairness threshold")


    # Get the arguments
    args = parser.parse_args()



    main(args)