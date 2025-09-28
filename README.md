# Crop-Recommendation-Project
# 🌾 **Crop Recommendation System**

## 📌 **Overview**

This project is a machine learning-based **crop recommendation system** that suggests the most suitable crop to grow based on environmental conditions like **NPK levels, temperature, humidity, pH, and rainfall**.

The system leverages machine learning and data-driven analysis to recommend crops that enhance agricultural yield while promoting long-term sustainability and resource efficiency. This helps farmers make informed decisions based on soil and climate conditions, ultimately improving productivity and supporting sustainable agriculture.

## ✨ **Features**

- Predicts the best crop based on soil and climate data.  
- Supports **multiple models** for comparison:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - LSTM (Deep Learning)  
  - CNN-LSTM (Deep Learning)  
- Displays accuracy and confusion matrix for each model.  
- Visual bar chart comparing model performance.

## ⚙️ **Requirements**

- Python 3.x  
- Libraries:  
  - `pandas`, `numpy`  
  - `matplotlib`, `seaborn`  
  - `scikit-learn`  
  - `keras`, `tensorflow`

Install required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## 📊 **Dataset**

The dataset used is `Crop_recommendation.csv` and contains the following features:

- **N**: Nitrogen  
- **P**: Phosphorus  
- **K**: Potassium  
- **temperature**: in °C  
- **humidity**: in %  
- **ph**: soil pH value  
- **rainfall**: in mm  
- **label**: crop name (target variable)

## 🚀 **Usage**

- Upload the dataset (`Crop_recommendation.csv`) in your Colab or local project.  
- Run the notebook step-by-step:  
  - Preprocess data (encoding, scaling)  
  - Train models  
  - Evaluate performance  
- View accuracy, confusion matrix, and model comparison bar chart.  
- Try changing model parameters to improve results.

## 🧐 **Model Implementation**

Before training the models, the dataset is preprocessed as follows:

- **Data Cleaning**: Ensured no missing or null values are present in the dataset.
- **Label Encoding**: The categorical crop labels are converted into numerical values using `LabelEncoder`.
- **Feature Normalization**: Input features are scaled using `StandardScaler` to bring them to a common scale for better model performance.
- **Train-Test Split**: The dataset is split into training and testing sets using an 80-20 ratio with `train_test_split(random_state=42)` to ensure consistent and reproducible evaluation results.

- **Logistic Regression**: Simple baseline model  
- **Decision Tree**: Rule-based classification  
- **Random Forest**: Ensemble of decision trees  
- **LSTM**: Deep learning model for sequential data  
- **CNN-LSTM**: Hybrid deep learning model combining CNN and LSTM layers

Each model is evaluated using:

- Accuracy  
- Confusion Matrix (normalized in %)  
- Classification Report (precision, recall, F1-score)

## 🔮 **Future Improvements**

- Develop a web interface using **Flask** or **Streamlit**.  
- Integrate real-time user input.  
- Experiment with more advanced models like **XGBoost** or **transformers**.  
- Deploy the system on the cloud (e.g., **Heroku**, **AWS**, or **Streamlit Cloud**).  
- Add location-based predictions using GPS and satellite data.

---

Feel free to fork this repo, explore the models, and contribute improvements!


🌱 Soil Moisture Prediction with LSTM
========================================

This project focuses on predicting soil moisture conditions using machine learning and deep learning techniques. 
It leverages time-series data, preprocessing, and an LSTM (Long Short-Term Memory) neural network for classification.

----------------------------------------
📌 Features
----------------------------------------
- Data preprocessing with pandas and scikit-learn
- Data visualization with matplotlib and seaborn
- Sequence generation for time-series modeling
- LSTM-based neural network using TensorFlow/Keras
- Model evaluation using accuracy, confusion matrix, and classification report

----------------------------------------
⚙ Requirements
----------------------------------------
Python >= 3.8

Libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `keras`
  
Install all with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow keras
```

----------------------------------------
📊 Model
----------------------------------------
- Two LSTM layers with 50 units each
- Dropout regularization
- Dense output layer with sigmoid activation
- Optimizer: Adam
- Loss: Binary Crossentropy

----------------------------------------
📈 Results
----------------------------------------
The notebook produces:
- Accuracy score
- Confusion matrix heatmap
- Classification report

----------------------------------------
🌍 Applications
----------------------------------------
- Smart irrigation systems
- Precision agriculture
- Drought monitoring
- Environmental sustainability

----------------------------------------
🔮 Future Work
----------------------------------------
- Integrate real-time IoT sensors for soil data collection
- Deploy as a web or mobile app for farmers
- Incorporate weather and satellite data for better prediction
- Add explainability (XAI) for transparency

Feel free to fork this repo, explore the models, and contribute improvements!


