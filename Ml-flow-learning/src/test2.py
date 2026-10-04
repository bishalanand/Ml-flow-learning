import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub


dagshub.init(repo_owner='bishalanand', repo_name='Ml-flow-learning', mlflow=True)

df=load_wine()
x=df.data
y=df.target 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=42)
max_depth=10
n_estimator=10
mlflow.set_tracking_uri("https://dagshub.com/bishalanand/Ml-flow-learning.mlflow")

# Create/select experiment
mlflow.set_experiment("Wine Classification")
with mlflow.start_run():
    rf=RandomForestClassifier(
        n_estimators=n_estimator,
        max_depth=max_depth,
        random_state=42
    )
    
    rf.fit(x_train,y_train)
    y_pred=rf.predict(x_test)
    
    accuracy=accuracy_score(y_pred,y_test)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('n_estimators',n_estimator)
    mlflow.log_param('max_depth',max_depth)
    mlflow.sklearn.log_model(rf,"Random Forest Model",skops_trusted_types=["sklearn.tree._tree.Tree"])
    mlflow.set_tags({
        "Created by":"Bishal",
        "for the":"Random forest"
    })
     