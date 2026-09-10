import numpy as np
import pandas as pd
import os
os.makedirs("data",exist_ok=True)
rng=np.random.default_rng(42)
diseases={"Flu":["fever","cough","fatigue","headache","body_ache","sore_throat","chills","runny_nose"],"Common Cold":["cough","sore_throat","runny_nose","sneezing","headache","fatigue"],"Migraine":["headache","nausea","vomiting","light_sensitivity","dizziness"],"Gastritis":["abdominal_pain","nausea","vomiting","loss_of_appetite","heartburn"],"Allergy":["sneezing","runny_nose","itchy_eyes","skin_rash","cough"],"Urinary Tract Infection":["painful_urination","frequent_urination","lower_abdominal_pain","fever","fatigue"],"Type 2 Diabetes":["excessive_thirst","frequent_urination","fatigue","blurred_vision","weight_loss"],"Hypertension":["headache","dizziness","blurred_vision","fatigue"],"Pneumonia":["fever","cough","shortness_of_breath","chest_pain","chills","fatigue"],"Asthma":["cough","shortness_of_breath","wheezing","chest_pain"]}
symptoms=sorted({s for v in diseases.values() for s in v}); rows=[]
for disease,core in diseases.items():
    for _ in range(120):
        r={s:int(rng.random()<(0.82 if s in core else 0.06)) for s in symptoms}
        if sum(r.values())<2:
            for s in rng.choice(core,size=min(2,len(core)),replace=False): r[s]=1
        r["disease"]=disease; rows.append(r)
pd.DataFrame(rows).sample(frac=1,random_state=42).to_csv("data/medical_symptoms.csv",index=False)
print("Synthetic dataset created.")
