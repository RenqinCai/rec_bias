import pandas as pd
import gurobipy as gp
import os
from gurobipy import GRB
from metric import get_ndcg, get_F1
import numpy as np

class UGF(object):
    def __init__(self, topk, epsilon, fairness_metric):
        self.m_epsilon = epsilon
        self.m_topk = topk
        self.m_name = "UGF"
        self.m_reranker = None
        self.m_fairness_metric = fairness_metric

    def f_train_fairness_reranker(self, group1_df, group2_df):
        self.m_reranker = gp.Model(self.m_name)

        obj_score_list = []  ### W_ij*S_ij
        group_fairness_metric_list = []

        group1_obj_score_list, group1_fairness_metric_list = self.f_proces_group(group1_df, self.m_fairness_metric)
        group2_obj_score_list, group2_fairness_metric_list = self.f_proces_group(group2_df, self.m_fairness_metric)

        obj_score_list.extend(group1_obj_score_list)
        obj_score_list.extend(group2_obj_score_list)

        group_fairness_metric_list.append(gp.quicksum(group1_fairness_metric_list)/len(group1_fairness_metric_list))
        group_fairness_metric_list.append(gp.quicksum(group2_fairness_metric_list)/len(group2_fairness_metric_list))

        self.m_reranker.update()

        self.m_reranker.addConstr(group_fairness_metric_list[0]-group_fairness_metric_list[1] <= self.m_epsilon)
        self.m_reranker.addConstr(group_fairness_metric_list[1]-group_fairness_metric_list[0] <= self.m_epsilon)

        self.m_reranker.setObjective(gp.quicksum(obj_score_list), GRB.MAXIMIZE)
        self.m_reranker.optimize()

    def f_proces_group(self, df, fairness_metric):
        user_groups = df.groupby("uid")

        tmp_obj_score_list = []
        tmp_fairness_metric_list = []
        
        for uid, group in user_groups:
            tmp_w_list = []
            tmp_w_label_list = []

            score_list = group["score"].tolist()
            label_list = group["label"].tolist()
            item_list = group["iid"].tolist()

            for i in range(len(item_list)):
                iid = item_list[i]
                w_name = str(uid)+"_"+str(iid)
                w = self.m_reranker.addVar(vtype=GRB.BINARY, name=w_name)

                tmp_w_list.append(w)
                tmp_w_label_list.append(label_list[i]*w)
                tmp_obj_score_list.append(score_list[i]*w)
                
            self.m_reranker.addConstr(gp.quicksum(tmp_w_list) == self.m_topk)

            if group["label"].sum() == 0:
                
                print("+++ no interacted items for this user +++")
                print("uid", uid)
                print("score_list", score_list)
                print("label_list", label_list)
                print("sum label", sum(label_list))
                print("item_list", item_list)
                exit()

            if fairness_metric == "f1":
                f1 = 2*gp.quicksum(tmp_w_label_list)/(group["label"].sum()+self.m_topk)
                tmp_fairness_metric_list.append(f1)

        return tmp_obj_score_list, tmp_fairness_metric_list

    def f_train(self, df, group1_df, group2_df):
        print("... start train the model ...")
        self.f_train_fairness_reranker(group1_df, group2_df)

        ### get train results
        W = self.m_reranker.getVars()

        wname_w_dict = {}

        for w in W:
            wname = w.varName

            wname_w_dict[wname] = int(w.x)
            # uid = int(w_name_list[0])
            # iid = int(w_name_list[1])

            # df.loc[(df["uid"]==uid) & (df["iid"]==iid), "w"] = int(w.x)

        uid_list = df["uid"].tolist()
        iid_list = df["iid"].tolist()
        w_list = []

        df_size = len(df)
        for i in range(df_size):
            uid_i = uid_list[i]
            iid_i = iid_list[i]
            wname = str(uid_i)+"_"+str(iid_i)
            w = wname_w_dict[wname]
            w_list.append(w)
        print("df_size", df_size, len(w_list))
        df.w = w_list
    
        group1_uid_list = df["uid"].tolist()
        group1_iid_list = df["iid"].tolist()
        group1_w_list = []

        group1_size = len(group1_df)
        for i in range(group1_size):
            uid_i = group1_uid_list[i]
            iid_i = group1_iid_list[i]
            wname = str(uid_i)+"_"+str(iid_i)
            w = wname_w_dict[wname]
            group1_w_list.append(w)

        group1_df.w = group1_w_list

        group2_uid_list = df["uid"].tolist()
        group2_iid_list = df["iid"].tolist()
        group2_w_list = []

        group2_size = len(group2_df)
        for i in range(group2_size):
            uid_i = group2_uid_list[i]
            iid_i = group2_iid_list[i]
            wname = str(uid_i)+"_"+str(iid_i)
            w = wname_w_dict[wname]
            group2_w_list.append(w)

        group2_df.w = group2_w_list

        # group1_df.drop(columns=["w"], inplace=True)
        # group1_df = pd.merge(group1_df, df, on=["uid", "iid", "score", "label"], how="left")

        # group2_df.drop(columns=["w"], inplace=True)
        # group2_df = pd.merge(group2_df, df, on=["uid", "iid", "score", "label"], how="left")

        ndcg, f1 = self.f_eval_df(df)
        ndcg_group1, f1_group1 = self.f_eval_df(group1_df)
        ndcg_group2, f1_group2 = self.f_eval_df(group2_df)

        print("all users ndcg:%.4f, f1:%.4f"%(ndcg, f1))
        print("group 1 ndcg:%.4f, f1:%.4f"%(ndcg_group1, f1_group1))
        print("group 2 ndcg:%.4f, f1:%.4f"%(ndcg_group2, f1_group2))
    
    def f_eval_df(self, df):
        w_list = df["w"].tolist()
        w = np.array(w_list)
        score_list = df["score"].tolist()
        score = np.array(score_list)

        w_s = w*score
        w_s = list(w_s)
        df["w_s"] = w_s
        # df["w_s"] = df["w"]*df["score"]

        # sorted_df = df.sort_values(by="w_s", ascending=True, ignore_index=True)
        df_groups = df.groupby("uid")

        topk = 20

        ndcg_list = []
        f1_list = []

        for uid, group in df_groups:
            label_list = group["label"].tolist()
            iid_list = group["iid"].tolist()
            w_score_list = group["w_s"].tolist()
            
            target_label_list, target_iid_list = zip(*sorted(zip(label_list, iid_list), reverse=True))
            pred_label_list, pred_iid_list = zip(*sorted(zip(w_score_list, iid_list), reverse=True))
            
            target_label_num = int(sum(target_label_list))
            targets = target_iid_list[:target_label_num]

            preds = pred_iid_list[:topk]

            print("preds list", pred_iid_list)
            print("preds", preds)
            print("targets", targets)

            ndcg = get_ndcg(preds, targets, topk)
            F1 = get_F1(preds, targets, topk)

            ndcg_list.append(ndcg)
            f1_list.append(F1)

            exit()

        mean_ndcg = np.mean(ndcg_list)
        mean_f1 = np.mean(f1_list)

        return mean_ndcg, mean_f1


            



        