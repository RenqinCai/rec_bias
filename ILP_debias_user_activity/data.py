import os
import pandas as pd
import json

class MOVIE(object):
    def __init__(self, pred_file, group_uid_file):
        self.m_eval_df = None
        
        self.m_group1_eval_df = None
        self.m_group2_eval_df = None

        self.f_load_pred_file(pred_file)
        self.f_load_group_file(group_uid_file)

    def f_load_pred_file(self, pred_file, sep="\t"):
        if os.path.exists(pred_file):
            if self.m_eval_df is None:
                print("... load pred file into eval df ...")
                self.m_eval_df = pd.read_pickle(pred_file)
                self.m_eval_df["w"] = 1

    
    def f_load_group_file(self, group_uid_file):
        if self.m_eval_df is None:
            print("first load pred file")

        group1_uid_list = [] 
        group2_uid_list = []

        bucketid_userid_dict_data = None
        with open(group_uid_file, "r") as f:
            bucketid_userid_dict_data = json.load(f)
            bucketid_userid_dict = bucketid_userid_dict_data["group_userid"]
        
        group1_uid_list = bucketid_userid_dict["1"]
        group2_uid_list = bucketid_userid_dict["2"]

        self.m_group1_eval_df = self.m_eval_df[self.m_eval_df["uid"].isin(group1_uid_list)]
        self.m_group2_eval_df = self.m_eval_df[self.m_eval_df["uid"].isin(group2_uid_list)]

        
