import joblib,pandas as pd
bundle=joblib.load("models/disease_prediction_model.joblib"); model=bundle["model"]; symptoms=bundle["symptoms"]
print("Educational demo only — NOT a medical diagnosis.\n")
selected=[]
for s in symptoms:
    if input(f"Do you have {s.replace('_',' ')}? (y/n): ").strip().lower()=="y": selected.append(s)
row=pd.DataFrame([[int(s in selected) for s in symptoms]],columns=symptoms)
pred=model.predict(row)[0]; probs=model.predict_proba(row)[0]
print("\nModel output (not a diagnosis):",pred)
for d,p in sorted(zip(model.classes_,probs),key=lambda z:z[1],reverse=True)[:3]: print(f"{d}: {p:.2%}")
