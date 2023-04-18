import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler


class FExtra:
    lr = LogisticRegression()
    robust_scaler = RobustScaler()
    num_embed = 0
    classes = []

    def __init__(self, features_train, num_embed, class_weight=None):
        self.num_embed = num_embed

        vec_train = []
        sign_train = []
        for edge in features_train:
            y = edge.vec[num_embed] if edge.vec[num_embed] == 1 else 0  # 1->1,-1->0
            edge_feature = np.array(edge.vec[0:num_embed])
            vec_train.append(edge_feature)
            sign_train.append(y)
        vec_train = np.array(vec_train)
        sign_train = np.array(sign_train)

        # for fitting scale
        self.robust_scaler = RobustScaler()
        self.robust_scaler.fit(vec_train)
        vec_train_robust_scaled = self.robust_scaler.transform(vec_train)

        # train LR and test
        self.lr = LogisticRegression(class_weight=class_weight, max_iter=1000)
        self.lr.fit(vec_train_robust_scaled, sign_train)
        self.classes = self.lr.classes_

    def compute_scores(self, features_test):
        vec_test = []
        for edge in features_test:
            edge_feature = np.array(edge.vec[0:self.num_embed])
            vec_test.append(edge_feature)
        vec_test = np.array(vec_test)

        self.robust_scaler.fit(vec_test)
        vec_test_robust_scaled = self.robust_scaler.transform(vec_test)

        # check
        test_score = self.lr.predict_proba(vec_test_robust_scaled)[:, 1]
        test_pred = self.lr.predict(vec_test_robust_scaled)
        
        return test_score, test_pred