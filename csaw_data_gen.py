import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csaw_df = pd.read_csv('demo/anon_dataset_nonhidden_211125.csv')


patients = csaw_df['anon_patientid'].unique()
print(len(patients))

patients_infos =[]
for p in patients:
    p_rows = csaw_df[csaw_df['anon_patientid']==p]
    patients_info = dict(
        anon_patientid = p,
        x_cancer_laterality = p_rows['x_cancer_laterality'].iloc[0],
        rad_timing = p_rows['rad_timing'].iloc[0]
    )
    years = p_rows['exam_year'].unique()
    for y in years:
        
        y_rows = p_rows[p_rows['exam_year']==y]
        year_info = dict(
            
        )

        print(len(y_rows))
