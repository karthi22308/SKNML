from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained KNN model
knn = joblib.load('knn_model.pkl')

# Define all possible areas
all_areas = ['Java', 'Git', 'SQL', 'Python', 'Machine Learning', 'Cloud Computing']


# Function to encode areas (strengths/weaknesses)
def encode_areas(areas, all_areas):
    return [1 if area in areas else 0 for area in all_areas]


# Course mapping logic
course_mapping = {
    'Java': ['Java for Beginners', 'Advanced Java'],
    'Git': ['Git for Beginners', 'Advanced Git'],
    'SQL': ['SQL for Beginners', 'Advanced SQL'],
    'Python': ['Python for Beginners', 'Advanced Python'],
    'Machine Learning': ['Intro to Machine Learning', 'Advanced Machine Learning'],
    'Cloud Computing': ['Intro to Cloud Computing', 'Cloud Security Basics'],
}


# Define the API route
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Parse input JSON
        data = request.json
        mcq_score = data['mcq_score']
        project_score = data['project_score']
        strength_areas = data['strength_areas']
        weak_areas = data['weak_areas']

        # Encode the input features
        candidate_features = np.array(
            [mcq_score, project_score] + encode_areas(strength_areas, all_areas) + encode_areas(weak_areas, all_areas)
        )

        # Get course recommendations based on KNN
        distances, indices = knn.kneighbors([candidate_features])

        # Load the original dataset to find course recommendations
        df = pd.read_excel("candidate_data.xlsx")

        recommended_courses = set()

        # Recommend advanced courses for strengths, beginner courses for weaknesses
        for idx in indices[0]:
            strengths = df.iloc[idx]['Strength_Areas'].split(', ')
            weaknesses = df.iloc[idx]['Weak_Areas'].split(', ')

            for area in strengths:
                if area not in weaknesses:
                    recommended_courses.add(course_mapping.get(area, [])[1])  # Advanced course

            for area in weaknesses:
                if area not in strengths:
                    recommended_courses.add(course_mapping.get(area, [])[0])  # Beginner course

        # Limit recommendations to 3 courses
        recommended_courses = list(recommended_courses)[:3]

        return jsonify({"recommended_courses": recommended_courses})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
