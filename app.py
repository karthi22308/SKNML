import streamlit as st
import numpy as np
import joblib
import pandas as pd
from openai import AzureOpenAI

def generate_response(input):
    client = AzureOpenAI(api_version='2024-06-01', azure_endpoint='https://hexavarsity-secureapi.azurewebsites.net/api/azureai', api_key='4ceeaa9071277c5b')
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': input}],
        temperature=0.7,
        max_tokens=2560,
        top_p=0.6,
        frequency_penalty=0.7
    )
    return res.choices[0].message.content
# Load the trained model
knn = joblib.load('knn_model.pkl')

# Define areas for strengths and weaknesses
all_areas = ['Java', 'Git', 'SQL', 'Python', 'Machine Learning', 'Cloud Computing']


# Function to encode areas (strengths/weaknesses)
def encode_areas(areas, all_areas):
    return [1 if area in areas else 0 for area in all_areas]


# Streamlit app layout
st.title("AI-Powered Course Recommendation")

# Input fields for MCQ Score, Project Score, Strength Areas, Weak Areas
mcq_score = st.slider("MCQ Score", min_value=50, max_value=100, value=75)
project_score = st.slider("Project Score", min_value=50, max_value=100, value=75)

strength_areas = st.multiselect(
    "Strength Areas",
    all_areas,
    default=["Java", "Python"]
)

weak_areas = st.multiselect(
    "Weak Areas",
    all_areas,
    default=["SQL"]
)

# Course mapping logic - Differentiate based on strength vs weakness
course_mapping = {
    'Java': ['Java for Beginners', 'Advanced Java'],
    'Git': ['Git for Beginners', 'Advanced Git'],
    'SQL': ['SQL for Beginners', 'Advanced SQL'],
    'Python': ['Python for Beginners', 'Advanced Python'],
    'Machine Learning': ['Intro to Machine Learning', 'Advanced Machine Learning'],
    'Cloud Computing': ['Intro to Cloud Computing', 'Cloud Security Basics'],
}

# When "Suggest" button is clicked
if st.button('Suggest Courses'):
    # Encode the input candidate features
    candidate_features = np.array(
        [mcq_score, project_score] + encode_areas(strength_areas, all_areas) + encode_areas(weak_areas, all_areas))

    # Get course recommendations based on KNN
    distances, indices = knn.kneighbors([candidate_features])

    # Load the original dataset to find course recommendations
    df = pd.read_excel("candidate_data.xlsx")

    recommended_courses = set()

    # For strengths, recommend advanced courses, for weaknesses recommend introductory courses
    for idx in indices[0]:
        strength_areas = df.iloc[idx]['Strength_Areas'].split(', ')
        weak_areas = df.iloc[idx]['Weak_Areas'].split(', ')

        # For each area in strength, recommend advanced courses (index 1)
        for area in strength_areas:
            if area not in weak_areas:
                recommended_courses.add(course_mapping.get(area, [])[1])  # Advanced course
        # For each area in weakness, recommend beginner courses (index 0)
        for area in weak_areas:
            if area not in strength_areas:
                recommended_courses.add(course_mapping.get(area, [])[0])  # Beginner course

    # Limit recommendations to 3 courses
    recommended_courses = list(recommended_courses)[:3]

    # Display the course recommendations
    st.write("Suggested Courses based on your input:")

    if recommended_courses:
        for course in recommended_courses:
            st.write(f"- {course}")
        prompt = 'you are course recommendation ai her is a candidate  with a strength areas :'+str(strength_areas)+'weak areas'+str(weak_areas)+'give some udemy course sussgestions and improvement plan for this candidtae'
        st.write("GenAi Suggestions:")
        st.write(generate_response(prompt))
    else:
        st.write("No course suggestions available based on the current input.")
