# 混合类型数据分类训练系统

一个基于Scikit-learn的完整机器学习流水线，专门用于处理混合类型（数值型+分类型）数据的分类问题。

## 🌟 核心特性

### 数据预处理
- **自动特征识别**：智能识别数值型和分类型特征
- **标准化处理**：数值特征标准化，分类特征独热编码
- **PCA降维**：可选的主成分分析降维功能

### 特征工程
- **包裹式特征选择**：支持前向/后向选择策略
- **可视化选择过程**：实时展示特征选择进度和效果
- **可配置参数**：自定义最小特征数量、评分标准等

### 类别不平衡处理
- **SMOTE过采样**：有效处理不平衡数据集
- **分布可视化**：对比展示采样前后类别分布
- **参数可调**：支持近邻数量等参数配置

### 模型训练与优化
- **多算法支持**：随机森林、支持向量机、逻辑回归
- **自动超参数调优**：基于网格搜索的参数优化
- **交叉验证**：可靠的模型性能评估

### 模型评估
- **全面评估指标**：准确率、F1分数、召回率等
- **可视化分析**：混淆矩阵、ROC曲线、特征重要性
- **交叉验证结果**：模型稳定性分析

## 🚀 快速开始

### 环境要求
```bash
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn
```

### 基本用法
```python
# 创建训练器实例
trainer = MixedTypeClassificationTrainer(
    use_pca=True,
    use_wrapper_selection=True,
    use_smote=True
)

# 加载数据
trainer.load_data(
    use_example=False,
    data_path="your_data.xlsx",
    target_column='target'
)

# 完整训练流程
trainer.train_baseline_models()\
      .optimize_best_model()\
      .evaluate_model()\
      .analyze_feature_importance()
```

### 高级配置
```python
# 自定义参数配置
wrapper_params = {
    'estimator': RandomForestClassifier(random_state=42),
    'scoring': 'recall',
    'direction': 'forward',
    'cv': 10,
    'min_features_to_select': 5
}

smote_params = {
    'k_neighbors': 5,
    'random_state': 42
}

trainer = MixedTypeClassificationTrainer(
    use_pca=True,
    pca_n_components=0.95,
    use_wrapper_selection=True,
    wrapper_params=wrapper_params,
    use_smote=True,
    smote_params=smote_params
)
```

## 📊 输出结果

系统提供丰富的输出信息：
- 数据预处理详情
- 特征选择过程可视化
- 模型性能比较
- 超参数优化结果
- 详细的评估报告和图表

## 🛠 核心类说明

### `MixedTypeClassificationTrainer`
主训练器类，管理整个机器学习流水线。

**主要方法：**
- `load_data()`: 加载和预处理数据
- `train_baseline_models()`: 训练基准模型
- `optimize_best_model()`: 优化最佳模型
- `evaluate_model()`: 全面评估模型
- `analyze_feature_importance()`: 特征重要性分析

### `WrapperFeatureSelector`
包裹式特征选择器，实现前向/后向特征选择算法。

## 📈 可视化功能

1. **特征选择过程**：展示特征数量与模型性能的关系
2. **类别分布**：SMOTE采样前后对比
3. **混淆矩阵**：模型预测结果可视化
4. **ROC曲线**：二分类问题性能评估
5. **特征重要性**：关键特征识别

## 💡 使用场景

- 🔍 **特征选择**：高维数据特征筛选
- ⚖️ **不平衡数据**：类别分布不均的数据集
- 📉 **维度灾难**：需要降维处理的数据
- 🔬 **模型比较**：多算法性能对比
- 🎯 **参数优化**：自动化超参数调优

## 🎯 优势特点

- **端到端解决方案**：从数据预处理到模型评估的完整流程
- **高度可配置**：灵活的参数设置适应不同需求
- **工业级代码**：完善的异常处理和进度监控
- **中文支持**：完整的中文界面和可视化
- **可扩展架构**：易于添加新的算法和功能

## 📝 注意事项

1. 确保数据文件路径正确
2. 目标变量应为数值型或可转换为数值型
3. 对于大规模数据，建议调整交叉验证折数
4. 可根据具体问题调整类别权重参数
