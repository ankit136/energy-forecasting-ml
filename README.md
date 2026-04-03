# ⚡ AI-Powered Energy Consumption Forecasting System

## 📌 Overview
This project predicts future energy consumption using Machine Learning.  
It uses time-series feature engineering like lag values and rolling averages to improve prediction accuracy.

---

## 🚀 Features
- Time-based features (hour, day, month, weekend)
- Lag feature and rolling mean
- Random Forest model
- Time-based train-test split
- Evaluation using RMSE and R² score
- Data visualization

---

## 🧠 How It Works
1. Generate or load data  
2. Clean the dataset  
3. Create new features (lag & rolling mean)  
4. Train the Random Forest model  
5. Predict energy consumption  
6. Evaluate model performance  
7. Save results in CSV file  

---

## 📂 Project Structure
energy-forecasting-ml/

- notebooks/project.ipynb → Full project with graphs  
- src/model.py → Main ML pipeline  
- outputs/predictions.csv → Model output  
- README.md  
- requirements.txt  

---

## 🛠 Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 📊 Results
- RMSE and R² used for evaluation  
- Model predicts energy trends effectively  
- Graphs show actual vs predicted values  

---

## ▶️ How to Run

Install dependencies:
pip install -r requirements.txt  

Run model:
python src/model.py  

---

## 💡 Use Cases
- Smart grid energy prediction  
- Industrial power management  
- Load forecasting  

---

## 📸 Project Output

(images/graph1.png)

(images/graph2.png)

