#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install geneticalgorithm')


# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
#import skfuzzy as fuzz
from sklearn.base import BaseEstimator, ClassifierMixin
from geneticalgorithm import geneticalgorithm as ga



# In[6]:


data=pd.read_csv('/content/postpartum_data.csv')


# In[7]:


data


# In[8]:


le=LabelEncoder()
y=le.fit_transform(data['RiskLevel'])


# In[9]:


y


# In[10]:


# Feature matrix and target
X = data.drop(columns=['RiskLevel'])


# In[11]:


X


# In[13]:


from sklearn.compose import ColumnTransformer
# 3. Identify column types **before splitting**
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 4. Split data (X_train remains a DataFrame)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Build preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])



# In[15]:


from sklearn.pipeline import Pipeline
models = {
    "SVC": SVC(),
    "RandomForest": RandomForestClassifier(random_state=42),
    "FFNN": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
}

# 8. Train and evaluate each model
for name, model in models.items():
    print(f"\n===== {name} =====")
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


# In[16]:


# 6. Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42))
])

# 7. Train model
pipeline.fit(X_train, y_train)

# 8. Predict and evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[17]:


import joblib

joblib.dump(le, "label_encoder.pkl")
joblib.dump(pipeline, "ppd_model_pipeline.pkl")


# In[ ]:


'''cols = ['Age', 'Q1', 'Q2','Q3','Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10','EPDS_Score' ,'FamilySupport']
X_test1 = [[25, 3, 3, 0, 1, 1, 0, 1, 2, 2, 2, 15, 'Medium']]
X_test1_df = pd.DataFrame(X_test1, columns=cols)

model.predict(X_test1_df)'''


# In[ ]:




