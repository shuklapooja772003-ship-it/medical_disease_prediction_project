import pandas as pd, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
os.makedirs("models",exist_ok=True)
df=pd.read_csv("data/medical_symptoms.csv"); symptoms=[c for c in df.columns if c!="disease"]
X=df[symptoms]; y=df.disease
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=250,random_state=42,n_jobs=-1)
model.fit(X_train,y_train); pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,pred)); print(classification_report(y_test,pred))
joblib.dump({"model":model,"symptoms":symptoms},"models/disease_prediction_model.joblib")
