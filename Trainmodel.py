import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load the data from the Excel file
df = pd.read_excel("candidate_data.xlsx")

# Encode strengths and weaknesses into binary vectors (1 or 0)
all_areas = ['Java', 'Git', 'SQL', 'Python', 'Machine Learning', 'Cloud Computing']

def encode_areas(areas, all_areas):
    return [1 if area in areas else 0 for area in all_areas]

# Encoding strength and weakness areas
df['Strength_Encoded'] = df['Strength_Areas'].apply(lambda x: encode_areas(x.split(', '), all_areas))
df['Weak_Encoded'] = df['Weak_Areas'].apply(lambda x: encode_areas(x.split(', '), all_areas))

# Combine features (MCQ Score, Project Score, and encoded strengths and weaknesses)
df['Features'] = df.apply(lambda row: [row['MCQ_Score'], row['Project_Score']] + row['Strength_Encoded'] + row['Weak_Encoded'], axis=1)

# Train the KNN model
X = np.array(df['Features'].tolist())
knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(X)

# Save the trained model (Optional, for future use)
import joblib
joblib.dump(knn, 'knn_model.pkl')
