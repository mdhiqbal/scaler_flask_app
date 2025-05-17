from flask import Flask, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("dt_model.pkl")

@app.route("/predict", methods=['POST'])
def prediction():
    app_req = request.get_json()

    # Grade category mapping
    grade_map = {
        'Grades PreK-2': 1,
        'Grades 3-5': 2,
        'Grades 6-8': 3,
        'Grades 9-12': 4
    }
    project_grade_category = grade_map.get(app_req["project_grade_category"], 0)

    # Essays length
    essay_1_len = len(app_req.get("project_essay_1", "").split())
    essay_2_len = len(app_req.get("project_essay_2", "").split())
    essay_3_len = len(app_req.get("project_essay_3", "").split())
    essay_4_len = len(app_req.get("project_essay_4", "").split())

    avg_essay_length = (essay_1_len + essay_2_len + essay_3_len + essay_4_len) / 4

    # Teacher experience
    prev_projects = app_req.get("teacher_number_of_previously_posted_projects", 0)
    teacher_experience = 1 if prev_projects > 0 else 0

    # Date parsing
    submitted_date = pd.to_datetime(app_req['project_submitted_datetime'])
    month_num = submitted_date.month
    weekday_num = submitted_date.weekday()

    # Subject category
    subject_category = app_req['project_subject_categories'].split(",")[0]

    subject_Health_Sports = int(subject_category == "Health & Sports")
    subject_History_Civics = int(subject_category == "History & Civics")
    subject_Literacy_Language = int(subject_category == "Literacy & Language")
    subject_Math_Science = int(subject_category == "Math & Science")
    subject_Music_The_Arts = int(subject_category == "Music & The Arts")
    subject_Special_Needs = int(subject_category == "Special Needs")

    # Prediction
    features = [[
        project_grade_category,
        prev_projects,
        essay_1_len,
        essay_2_len,
        essay_3_len,
        essay_4_len,
        teacher_experience,
        month_num,
        weekday_num,
        avg_essay_length,
        subject_Health_Sports,
        subject_History_Civics,
        subject_Literacy_Language,
        subject_Math_Science,
        subject_Music_The_Arts,
        subject_Special_Needs
    ]]

    result = model.predict(features)

    pred = "Approved" if result[0] == 1 else "Rejected"
    return {"Application_status": pred}

if __name__ == "__main__":
    app.run(debug=True)
