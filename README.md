# 🏠 House Price Prediction

The real estate industry often needs accurate estimation of property prices based on various house characteristics. This project builds a Machine Learning regression model that predicts the price of a house using features such as house size, number of bedrooms, number of bathrooms, and the year the house was built.

## 📁 Project Structure

```
House-Price-ML/
│
├── notebook.ipynb        # Colab/Jupyter notebook with full ML workflow
├── app.py                # Streamlit web app for interactive predictions
├── model.pkl             # Trained Linear Regression model
├── scaler.pkl            # Fitted StandardScaler for feature normalization
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```bash
streamlit run app.py
```

### 3. Open the notebook
Open `notebook.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/) to explore the full training pipeline.

## 🔍 Features Used

| Feature      | Description                  |
|-------------|------------------------------|
| `sqft`      | House size in square feet    |
| `bedrooms`  | Number of bedrooms           |
| `bathrooms` | Number of bathrooms          |
| `year_built`| Year the house was built     |

## 📊 Model

- **Algorithm:** Linear Regression  
- **Preprocessing:** StandardScaler  
- **Evaluation Metrics:** MAE, RMSE, R²
