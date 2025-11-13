import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc, f1_score)
from sklearn.exceptions import NotFittedError
from imblearn.over_sampling import SMOTE  
from collections import Counter  
import numpy as np

# 中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 包裹式选择类
class WrapperFeatureSelector:
    def __init__(self, estimator, scoring="f1", cv=5, direction="forward", min_features_to_select=1):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.direction = direction
        self.min_features_to_select = min_features_to_select
        self.selected_feature_indices = [] 
        self.scores = []  
        self.feature_history = []

    def _evaluate(self, X, y, feature_idx):
        X_subset = X[:, feature_idx] if isinstance(X, np.ndarray) else X.iloc[:, feature_idx]
        try:
            scores = cross_val_score(self.estimator, X_subset, y, scoring=self.scoring, cv=self.cv)
            return np.mean(scores)
        except NotFittedError:
            return -np.inf

    def fit(self, X, y):
        n_features = X.shape[1]
        if self.direction == "forward":
            current_features = []
            remaining_features = list(range(n_features))
        else:
            current_features = list(range(n_features))
            remaining_features = []

        while True:
            best_score = -np.inf
            best_feature = None

            if self.direction == "forward":
                for feat in remaining_features:
                    candidate = current_features + [feat]
                    current_score = self._evaluate(X, y, candidate)
                    if current_score > best_score:
                        best_score = current_score
                        best_feature = feat
                
                if best_feature is None:
                    # 如果还没达到最小特征数量，强制选择剩余特征中评分最高的
                    if len(current_features) < self.min_features_to_select and remaining_features:
                        # 选择评分最高的剩余特征
                        best_feature = max(remaining_features, 
                                          key=lambda f: self._evaluate(X, y, current_features + [f]))
                        best_score = self._evaluate(X, y, current_features + [best_feature])
                    else:
                        break
                elif best_score <= (self.scores[-1] if self.scores else -np.inf) and len(current_features) >= self.min_features_to_select:
                    break
                
                current_features.append(best_feature)
                remaining_features.remove(best_feature)
            else:  # 后向选择
                for feat in current_features:
                    candidate = [f for f in current_features if f != feat]
                    current_score = self._evaluate(X, y, candidate)
                    if current_score > best_score:
                        best_score = current_score
                        best_feature = feat
                
                # 后向选择的停止条件：没有更好的特征可移除，或者达到最小特征数量
                if best_feature is None or len(current_features) - 1 < self.min_features_to_select:
                    break
                
                current_features.remove(best_feature)
                remaining_features.append(best_feature)

            self.scores.append(best_score)
            self.feature_history.append(current_features.copy())
            print(f"包裹式选择步骤{len(self.scores)}: 选中特征数={len(current_features)}, 最优{self.scoring}={best_score:.4f}")

        # 确保最终特征数量不小于最小要求
        if len(current_features) < self.min_features_to_select:
            print(f"警告：无法达到最小特征数量要求，已选择所有可用特征（{len(current_features)}个）")
        
        self.selected_feature_indices = current_features
        return self

    def transform(self, X):
        return X[:, self.selected_feature_indices] if isinstance(X, np.ndarray) else X.iloc[:, self.selected_feature_indices]

    def plot_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(
            [len(feats) for feats in self.feature_history],
            self.scores,
            marker="o", linestyle="-", color="#2E86AB"
        )
        plt.xlabel("选中的特征数量", fontsize=12)
        plt.ylabel(f"交叉验证{self.scoring}分数", fontsize=12)
        plt.title(f"包裹式特征选择过程（{self.direction}）", fontsize=14, fontweight="bold")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

class MixedTypeClassificationTrainer:
    def __init__(self, use_pca=False, pca_n_components=0.95, 
                 use_wrapper_selection=False, wrapper_params=None,
                 use_smote=False, smote_params=None): 
        self.models = {
            '随机森林': RandomForestClassifier(random_state=42),
            '支持向量机': SVC(probability=True, random_state=42),
            '逻辑回归': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.best_model = None
        self.preprocessor = None
        self.categorical_features = None
        self.numerical_features = None
        self.use_pca = use_pca
        self.pca_n_components = pca_n_components
        self.pca = None
        
        self.use_wrapper_selection = use_wrapper_selection
        self.wrapper_selector = None
        self.wrapper_params = wrapper_params if wrapper_params else {}
        
        self.use_smote = use_smote
        self.smote = None  
        self.smote_params = smote_params if smote_params else {}  
        self.X_train_resampled = None  

    def load_data(self, use_example=True, data_path=None, target_column=None):
        if use_example:
            titanic = fetch_openml('titanic', version=1, as_frame=True).frame
            self.df = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'survived']].dropna()
            self.target_column = 'survived'
            self.X = self.df.drop(columns=[self.target_column])
            self.y = self.df[self.target_column].astype(int)
            self.categorical_features = ['pclass', 'sex']
            self.numerical_features = ['age', 'sibsp', 'parch', 'fare']
            print("已加载示例泰坦尼克号数据集（混合类型）")
        else:
            if not data_path or not target_column:
                raise ValueError("使用自定义数据时，必须指定data_path和target_column")
            self.df = pd.read_excel(data_path)
            self.target_column = target_column
            self.X = self.df.drop(columns=[self.target_column])
            self.y = self.df[self.target_column]
            self.categorical_features = []
            self.numerical_features = []
            for col in self.X.columns:
                if self.X[col].dtype == 'object':
                    self.categorical_features.append(col)
                elif self.X[col].dtype in ['int64', 'int32'] and self.X[col].nunique() <= 10:
                    self.categorical_features.append(col)
                else:
                    self.numerical_features.append(col)
            print(f"自动识别特征类型：")
            print(f"分类变量 ({len(self.categorical_features)}个): {self.categorical_features}")
            print(f"连续变量 ({len(self.numerical_features)}个): {self.numerical_features}")
        
        print(f"\n原始数据类别分布: {dict(Counter(self.y))}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"训练集原始类别分布: {dict(Counter(self.y_train))}")
        
        self._build_preprocessor()
        
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        if self.use_pca:
            pre_pca_dim = self.preprocessor.named_steps['pre_pca'].transform(self.X.iloc[:10]).shape[1]
            post_pca_dim = self.X_train_processed.shape[1]
            print(f"\nPCA降维效果：")
            print(f"降维前特征维度：{pre_pca_dim} → 降维后：{post_pca_dim}")
            self.pca = self.preprocessor.named_steps['pca']
        else:
            print(f"\n预处理后特征维度（无PCA）：{self.X_train_processed.shape[1]}")
        
        if self.use_wrapper_selection:
            self._run_wrapper_selection()
        
        if self.use_smote:
            self._run_smote()
            self.X_train_used = self.X_train_resampled
        else:
            self.X_train_used = self.X_train_processed if not self.use_wrapper_selection else self.X_train_selected
        
        self.X_test_used = self.X_test_processed if not self.use_wrapper_selection else self.X_test_selected
        
        return self

    def _build_preprocessor(self):
        numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        pre_pca_transformer = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        if self.use_pca:
            self.preprocessor = Pipeline(steps=[
                ('pre_pca', pre_pca_transformer),
                ('pca', PCA(n_components=self.pca_n_components, random_state=42))
            ])
            print(f"已启用PCA降维，参数：n_components={self.pca_n_components}")
        else:
            self.preprocessor = pre_pca_transformer
            print("未启用PCA降维，使用原始编码+标准化特征")
        
        return self

    def _run_wrapper_selection(self):
        print("\n===== 开始包裹式特征选择 =====")
        wrapper_estimator = self.wrapper_params.get('estimator', RandomForestClassifier(random_state=42))
        wrapper_scoring = self.wrapper_params.get('scoring', 'f1')
        wrapper_direction = self.wrapper_params.get('direction', 'forward')
        wrapper_cv = self.wrapper_params.get('cv', 5)
        wrapper_num = self.wrapper_params.get('min_features_to_select', 1)
        
        self.wrapper_selector = WrapperFeatureSelector(
            estimator=wrapper_estimator,
            scoring=wrapper_scoring,
            direction=wrapper_direction,
            cv=wrapper_cv,
            min_features_to_select=wrapper_num
        )
        
        self.wrapper_selector.fit(self.X_train_processed, self.y_train)
        
        self.X_train_selected = self.wrapper_selector.transform(self.X_train_processed)
        self.X_test_selected = self.wrapper_selector.transform(self.X_test_processed)
        
        print(f"\n包裹式选择完成：")
        print(f"筛选前特征数：{self.X_train_processed.shape[1]} → 筛选后：{self.X_train_selected.shape[1]}")
        print(f"最优{wrapper_scoring}分数：{max(self.wrapper_selector.scores) if self.wrapper_selector.scores else 'N/A'}")
        self.wrapper_selector.plot_history()

    def _run_smote(self):
        print("\n===== 开始SMOTE过采样 =====")
        smote_k_neighbors = self.smote_params.get('k_neighbors', 5)
        smote_random_state = self.smote_params.get('random_state', 42)
        
        self.smote = SMOTE(k_neighbors=smote_k_neighbors, random_state=smote_random_state)
        
        X_train_for_smote = self.X_train_selected if self.use_wrapper_selection else self.X_train_processed
        
        self.X_train_resampled, self.y_train_resampled = self.smote.fit_resample(
            X_train_for_smote, self.y_train
        )
        
        print(f"SMOTE前训练集类别分布: {dict(Counter(self.y_train))}")
        print(f"SMOTE后训练集类别分布: {dict(Counter(self.y_train_resampled))}")
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.countplot(x=self.y_train)
        plt.title('SMOTE前训练集类别分布')
        plt.subplot(1, 2, 2)
        sns.countplot(x=self.y_train_resampled)
        plt.title('SMOTE后训练集类别分布')
        plt.tight_layout()
        plt.show()

    def train_baseline_models(self):
        print("\n===== 训练基准模型 =====")
        self.baseline_results = {}
        
        y_train_used = self.y_train_resampled if self.use_smote else self.y_train
        
        for name, model in self.models.items():
            model.fit(self.X_train_used, y_train_used) 
            
            y_pred = model.predict(self.X_test_used)
            y_pred_proba = model.predict_proba(self.X_test_used)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            
            self.baseline_results[name] = {
                'model': model, 'accuracy': accuracy, 'report': report,
                'y_pred': y_pred, 'y_pred_proba': y_pred_proba
            }
            
            print(f"\n{name} 准确率: {accuracy:.4f}")
            print(f"{name} 分类报告:\n{report}")
        
        best_model_name = max(self.baseline_results.items(), key=lambda x: x[1]['accuracy'])[0]
        self.best_model = self.baseline_results[best_model_name]['model']
        print(f"\n初步最佳模型: {best_model_name}")
        
        return self

    def optimize_best_model(self):
        if self.best_model is None:
            raise ValueError("请先训练基准模型")
            
        print("\n===== 优化最佳模型 =====")
        model_class_name = self.best_model.__class__.__name__
        
        param_grids = {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'class_weight':[{0:1, 1:1.8}]
            },
            'SVC': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 0.1,0.01],
                'kernel': ['linear', 'rbf'],
                'class_weight':[{0:1, 1:2}]
            },
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear'],
                'class_weight':[{0:1, 1:3}]
            }
        }
        
        param_grid = param_grids.get(model_class_name, {})
        if not param_grid:
            print("没有为该模型定义超参数网格，使用默认参数")
            return self
        
        X_train_used = self.X_train_used
        y_train_used = self.y_train_resampled if self.use_smote else self.y_train
        
        grid_search = GridSearchCV(
            estimator=self.best_model, param_grid=param_grid,
            cv=5, n_jobs=-1, verbose=1, scoring='f1'
        )
        grid_search.fit(X_train_used, y_train_used)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        self.best_model = grid_search.best_estimator_
        
        y_pred = self.best_model.predict(self.X_test_used)
        print(f"优化后模型在测试集上的准确率: {accuracy_score(self.y_test, y_pred):.4f}")
        
        return self

    def evaluate_model(self):
        if self.best_model is None:
            raise ValueError("请先训练模型")
            
        print("\n===== 最佳模型详细评估 =====")
        y_pred = self.best_model.predict(self.X_test_used)
        y_pred_proba = self.best_model.predict_proba(self.X_test_used)
        
        print("\n分类报告:")
        print(classification_report(self.y_test, y_pred))
        
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(self.y),
                   yticklabels=np.unique(self.y))
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.show()
        
        X_train_used = self.X_train_used
        y_train_used = self.y_train_resampled if self.use_smote else self.y_train
        
        cv_scores = cross_val_score(self.best_model, X_train_used, y_train_used, cv=5)
        print(f"\n交叉验证分数: {cv_scores}")
        print(f"交叉验证平均分数: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        if len(np.unique(self.y)) == 2:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlabel('假正例率')
            plt.ylabel('真正例率')
            plt.title('ROC曲线')
            plt.legend(loc="lower right")
            plt.show()
        
        return self

    def analyze_feature_importance(self):
        if self.best_model is None:
            raise ValueError("请先训练模型")
            
        model = self.best_model
        if not hasattr(model, 'feature_importances_'):
            print("\n当前模型不支持特征重要性分析（仅树模型支持）")
            return self
        
        print("\n===== 特征重要性分析 =====")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        if self.use_wrapper_selection and self.wrapper_selector:
            selected_indices = self.wrapper_selector.selected_feature_indices
        else:
            selected_indices = list(range(self.X_train_used.shape[1]))
        
        if self.use_pca:
            feature_names = [f'主成分{i+1}' for i in selected_indices]
            plt.figure(figsize=(12, 6))
            plt.bar(range(min(10, len(indices))), importances[indices[:10]])
            plt.xticks(range(min(10, len(indices))), [feature_names[i] for i in indices[:10]], rotation=90)
            plt.xlabel('PCA主成分')
            plt.ylabel('重要性')
            plt.title('特征重要性（前10名）')
            plt.tight_layout()
            plt.show()
            
            print("\n主成分重要性排序（前10名）:")
            for f in range(min(10, len(indices))):
                pc_idx = selected_indices[indices[f]]
                var_ratio = self.pca.explained_variance_ratio_[pc_idx]
                print(f"{feature_names[indices[f]]}: 重要性={importances[indices[f]]:.4f}, 解释方差={var_ratio:.4f}")
        else:
            numerical_names = self.numerical_features.copy()
            ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_ohe_names = list(ohe.get_feature_names_out(self.categorical_features))
            all_feature_names = numerical_names + cat_ohe_names
            
            if self.use_wrapper_selection:
                feature_names = [all_feature_names[i] for i in selected_indices]
            else:
                feature_names = all_feature_names
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(min(10, len(indices))), importances[indices[:10]])
            plt.xticks(range(min(10, len(indices))), [feature_names[i] for i in indices[:10]], rotation=90)
            plt.xlabel('特征')
            plt.ylabel('重要性')
            plt.title('特征重要性（前10名）')
            plt.tight_layout()
            plt.show()
            
            print("\n特征重要性排序（前10名）:")
            for f in range(min(10, len(indices))):
                print(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")\
        
        return self

    def predict_new_data(self, new_data):
        if self.best_model is None:
            raise ValueError("请先训练并优化模型")
            
        new_data_processed = self.preprocessor.transform(new_data)
        if self.use_wrapper_selection and self.wrapper_selector:
            new_data_processed = self.wrapper_selector.transform(new_data_processed)
            
        predictions = self.best_model.predict(new_data_processed)
        pred_proba = self.best_model.predict_proba(new_data_processed)
        
        return predictions, pred_proba


def main():
    # 配置包裹式特征选择参数
    wrapper_params = {
        'estimator': RandomForestClassifier(random_state=42),
        'scoring': 'recall', # 评分依据
        'direction': 'forward', # 筛选方向
        'cv': 10, # 交叉验证折数
        'min_features_to_select':5 # 最少特征数
    }
    
    # 配置SMOTE参数
    smote_params = {
        'k_neighbors': 5,  # 近邻数量
        'random_state': 42 # 随机种子
    }
    
    # 创建训练器
    trainer = MixedTypeClassificationTrainer(
        use_pca=True, # 是否降维
        pca_n_components=0.95, # 降维时的解释方差阈值
        use_wrapper_selection=True, # 是否包裹式选择特征
        wrapper_params=wrapper_params, # 包裹参数
        use_smote=True,  # 启用SMOTE过采样
        smote_params=smote_params
    )
    
    # 加载数据
    trainer.load_data(
        use_example=False,
        data_path=r"",
        target_column=''
    )
    
    # 训练基准模型
    trainer.train_baseline_models()
    
    # 优化最佳模型
    trainer.optimize_best_model()
    
    # 评估模型
    trainer.evaluate_model()
    
    # 分析特征重要性
    trainer.analyze_feature_importance()
    
    print("\n模型训练和评估完成！")

if __name__ == "__main__":
    main()
