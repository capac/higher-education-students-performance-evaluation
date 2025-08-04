import requests

student = {
    "Student Age": 2, "Sex": 1, "Graduated high-school type": 2,
    "Scholarship type": 1, "Additional work": 2,
    "Regular artistic or sports activity": 2, "Do you have a partner": 2,
    "Total salary if available": 3, "Transportation to the university": 1,
    "Accommodation type in Cyprus": 1, "Mother's education": 4,
    "Father's education": 3, "Number of sisters/brothers (if available)": 2,
    "Parental status": 1, "Mother's occupation": 3, "Father's occupation": 3,
    "Weekly study hours": 2,
    "Reading frequency (non-scientific books/journals)": 1,
    "Reading frequency (scientific books/journals)": 2,
    "Attendance to the seminars/conferences related to the department": 1,
    "Impact of your projects/activities on your success": 1,
    "Attendance to classes": 1, "Preparation to midterm exams 1": 1,
    "Preparation to midterm exams 2": 2, "Taking notes in classes": 3,
    "Listening in classes": 2,
    "Discussion improves my interest and success in the course": 2,
    "Flip-classroom": 2,
    "Cumulative grade point average in the last semester (/4.00)": 4,
    "Expected Cumulative grade point average in the graduation (/4.00)": 4,
    "COURSE ID": 1}

url = "http://localhost:9696/predict"
response = requests.post(url, json=student)
print(response.json())
# print("Status code:", response.status_code)
# print("Response text:", response.text)
