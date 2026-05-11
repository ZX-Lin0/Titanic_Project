# Titanic Survival Prediction API

基于泰坦尼克号乘客数据，使用随机森林算法构建生存预测模型，并通过Flask部署为RESTful API。

## 技术栈
- **数据预处理**: Pandas
- **模型**: Scikit-learn (RandomForestClassifier)
- **API部署**: Flask
- **序列化**: Joblib

## 项目结构
| 文件 | 功能 |
|------|------|
| `train.py` | 数据加载、缺失值填补、特征转换、模型训练，保存模型 |
| `predict.py` | 加载模型，命令行预测工具 |
| `app.py` | Flask API服务，提供RESTful接口 |
| `data/` | 训练与测试数据集 |
| `model.pkl` | 训练好的随机森林模型 |

## 快速开始

### 1. 安装依赖
`pip install pandas scikit-learn flask joblib`

### 2. 训练模型
`python train.py`

### 3. 启动API服务
`python app.py`

### 4. 调用API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[3, 1, 25, 0, 0, 7.25]}'

### 返回示例
{
  "survived": 1
}

## 后续改进方向
- 增加模型评估指标（准确率、混淆矩阵）
- 部署到云服务器
- 添加前端交互页面
