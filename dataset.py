import pandas as pd
import random

# Define areas for strengths and weaknesses
all_areas = ['Java', 'Git', 'SQL', 'Python', 'Machine Learning', 'Cloud Computing']


# Function to create sample data
def create_sample_data():
    data = {
        'Candidate_ID': [f'C{i}' for i in range(1, 21)],
        'MCQ_Score': [random.randint(50, 100) for _ in range(20)],
        'Project_Score': [random.randint(50, 100) for _ in range(20)],
        'Strength_Areas': [', '.join(random.sample(all_areas, 3)) for _ in range(20)],
        'Weak_Areas': [', '.join(random.sample(all_areas, 2)) for _ in range(20)],
    }

    df = pd.DataFrame(data)

    # Save to Excel or CSV
    df.to_excel("candidate_data.xlsx", index=False)  # Save as Excel file
    # df.to_csv("candidate_data.csv", index=False)  # Save as CSV file

    return df


# Generate the dummy data and save it as Excel
create_sample_data()
