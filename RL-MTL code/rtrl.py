import numpy as np
import pandas as pd
import torch
from collections import defaultdict

class RetailRocketRLDataset(torch.utils.data.Dataset):
    """
    RetailRocketRec Dataset
    """
    def __init__(self, dataset_path):

        self.cate_cols = ['785', '591', '814', 'available', 'categoryid', '364', '776']
        self.filter_cols = ['776', '364']
        self.features = self.get_features("./dataset/rt/item_feadf.csv", self.cate_cols)
        mdp_data = pd.read_csv(dataset_path, usecols=["visitorid", 'itemid', 'click', 'pay'],engine='python',error_bad_lines=False)
        data = mdp_data.merge(self.features, on="itemid", how='left')
        data.fillna(0)
        self.categorical_data = data[self.cate_cols].values.astype(np.int)

        self.field_dims = np.max(self.categorical_data, axis=0) + 1
        # add 0 numerical_data
        self.numerical_data = np.zeros((data.shape[0],1)).astype(np.float32)
        self.numerical_num = self.numerical_data.shape[1]
        self.labels = data[['click', 'pay']].values.astype(np.float32)
        self.session_id = mdp_data['visitorid'].values.astype(np.int)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        cate_item = self.categorical_data[index]
        label_item = self.labels[index]
        num_item = self.numerical_data[index]
        session_id = self.session_id[index]
        return session_id, cate_item, num_item, label_item

    def get_labels(self):
        return np.where(np.sum(self.labels,axis=1)>0,1,0)

    def get_features(self, features_path,feature_cols):
        features = pd.read_csv(features_path, usecols=feature_cols + ['itemid'])
        features.drop_duplicates('itemid', inplace=True)
        features.fillna(0, inplace=True)
        return features


if __name__ == '__main__':
    datapath = "./rt/test.csv"
    db = RetailRocketRLDataset(dataset_path=datapath)
