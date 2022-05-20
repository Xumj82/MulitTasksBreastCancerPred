
def get_exam_level_meta(csaw_df):
    csaw_df_exam_level = csaw_df[['anon_patientid',	
                'exam_year',
                'x_age',	
                'x_case',	
                'x_cancer_laterality',	
                'x_type',	
                'x_lymphnode_met',	
                'rad_timing',	
                'rad_r1',	
                'rad_r2',	
                'rad_recall',	
                'rad_recall_type_right',	
                'rad_recall_type_left'
                ]].groupby(['anon_patientid','exam_year']).agg(
                x_case = ('x_case', 'max'),
                x_age = ('x_age','max'),
                x_cancer_laterality = ('x_cancer_laterality','max'),
                x_type = ('x_type','min'),
                x_lymphnode_met = ('x_lymphnode_met','max'),
                rad_timing = ('rad_timing','min'),
                rad_r1 = ('rad_r1','max'),
                rad_r2 = ('rad_r2','max'),
                rad_recall = ('rad_recall','max'),
                rad_recall_type_right = ('rad_recall_type_right','max'),
                rad_recall_type_left = ('rad_recall_type_left','max')
                ).reset_index()
    return csaw_df_exam_level

def get_patient_level_meta(csaw_df):
    csaw_df_exam_level = get_exam_level_meta(csaw_df)
    csaw_df_patient_level = csaw_df_exam_level[['anon_patientid',
                'exam_year',
                'x_age',
                'x_case',
                'x_cancer_laterality',
                'rad_timing']].groupby(['anon_patientid']).agg(
                    x_case = ('x_case', 'max'),
                    x_age = ('x_age','max'),
                    x_cancer_laterality = ('x_cancer_laterality','max'),
                    rad_timing = ('rad_timing','min'),
                    year_count = ('exam_year','count'),
                    max_year = ('exam_year','max'),
                    min_year = ('exam_year','min'),
                    ).reset_index()
    csaw_df_patient_level['duration'] = csaw_df_patient_level['max_year']-csaw_df_patient_level['min_year']+1
    return csaw_df_patient_level