import streamlit as st
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
import io
from pdfminer.high_level import extract_text
import requests
import time
import mysql.connector
from datetime import datetime, timedelta

try:
    from langchain.llms import OpenAI
except ImportError:
    OpenAI = None

MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DATABASE = "job_ads"

def create_database_and_tables():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
    cursor.execute(f"USE {MYSQL_DATABASE}")
    cursor.execute('''CREATE TABLE IF NOT EXISTS advertisements (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        company_name VARCHAR(255),
                        job_title VARCHAR(255),
                        advertisement TEXT,
                        published BOOLEAN DEFAULT FALSE,
                        start_time DATETIME,
                        expire_time DATETIME
                      )''')
    conn.commit()
    cursor.close()
    conn.close()

create_database_and_tables()

def save_advertisement(company_name, job_title, advertisement, published=False):
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    cursor = conn.cursor()

    # Check if there's an active advertisement for the company
    cursor.execute('''SELECT COUNT(*) FROM advertisements 
                      WHERE company_name = %s AND published = TRUE AND expire_time > NOW()''', 
                   (company_name,))
    active_ads = cursor.fetchone()[0]

    if active_ads > 0:
        raise Exception("Cannot publish more than one advertisement at a time for the same company.")

    start_time = datetime.now()
    expire_time = start_time + timedelta(days=3)

    cursor.execute('''INSERT INTO advertisements (company_name, job_title, advertisement, published, start_time, expire_time)
                      VALUES (%s, %s, %s, %s, %s, %s)''',
                   (company_name, job_title, advertisement, published, start_time, expire_time))
    conn.commit()
    cursor.close()
    conn.close()

def get_advertisements(published=None):
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    cursor = conn.cursor(dictionary=True)
    if published is None:
        cursor.execute("SELECT * FROM advertisements")
    else:
        cursor.execute("SELECT * FROM advertisements WHERE published = %s", (published,))
    ads = cursor.fetchall()
    cursor.close()
    conn.close()
    return ads

class AdvertisementError(Exception):
    pass

def publish_advertisement(company_name, ad_id):
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        cursor = conn.cursor()

        # Check if there's an active advertisement for the company
        cursor.execute('''SELECT COUNT(*) FROM advertisements 
                          WHERE company_name = %s AND published = TRUE AND expire_time > NOW()''', 
                       (company_name,))
        active_ads = cursor.fetchone()[0]

        if active_ads > 0:
            raise AdvertisementError("Cannot publish more than one advertisement at a time for the same company. The current advertisement will expire 3 days after creation, at which point you can upload a new one.")

        start_time = datetime.now()
        expire_time = start_time + timedelta(days=3)

        cursor.execute('''UPDATE advertisements 
                          SET published = TRUE, start_time = %s, expire_time = %s 
                          WHERE id = %s''', 
                       (start_time, expire_time, ad_id))
        conn.commit()
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise Exception("An error occurred while updating the advertisement.")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def generate_code(profile):
    backend_code = '''
from fastapi import FastAPI, HTTPException, Depends, Security, Body
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import List
import base64
import io
from pdfminer.high_level import extract_text
import json
import databases
import sqlalchemy
from sqlalchemy import UniqueConstraint

DATABASE_URL = "sqlite:///./test.db"
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("email", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("mobile", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("cv_base64", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("score", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("feedback", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("skills_matched", sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("total_skills", sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("total_priority", sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("matched_priority", sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("passed", sqlalchemy.Boolean, nullable=False),
    UniqueConstraint("email", name="uq_user_email")
)

engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)

app = FastAPI()

security = HTTPBasic()

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"

class CVRequest(BaseModel):
    pdf_base64: str
    name: str
    email: str
    mobile: str

class EvaluationResult(BaseModel):
    score: float
    feedback: str
    extracted_text: str
    skills_matched: int
    total_skills: int
    total_priority: int
    matched_priority: int
    passed: bool

class Skill(BaseModel):
    name: str
    priority: int

class JobProfile(BaseModel):
    job_title: str
    skills: List[Skill]
    pass_threshold: float

job_profile = JobProfile(**json.loads(''' + json.dumps(json.dumps(profile)) + '''))

def extract_text_from_base64(pdf_base64: str) -> str:
    pdf_bytes = base64.b64decode(pdf_base64)
    text = extract_text(io.BytesIO(pdf_bytes))
    return text

def evaluate_cv(text: str, job_profile: JobProfile) -> EvaluationResult:
    skills = job_profile.skills
    normalized_text = text.lower()

    matched_skills = [skill for skill in skills if skill.name.lower() in normalized_text]
    total_priority = sum(skill.priority for skill in skills)
    matched_priority = sum(skill.priority for skill in matched_skills)

    score = (matched_priority / total_priority) * 100 if total_priority > 0 else 0
    passed = score >= job_profile.pass_threshold

    feedback = "CV evaluated successfully." if matched_priority > 0 else "No relevant skills found in CV."

    return EvaluationResult(
        score=score,
        feedback=feedback,
        extracted_text=text,
        skills_matched=len(matched_skills),
        total_skills=len(skills),
        total_priority=total_priority,
        matched_priority=matched_priority,
        passed=passed
    )

@app.post("/api/evaluate_cv", response_model=EvaluationResult)
async def evaluate_cv_endpoint(cv_request: CVRequest):
    text = extract_text_from_base64(cv_request.pdf_base64)
    result = evaluate_cv(text, job_profile)

    query = users.insert().values(
        name=cv_request.name,
        email=cv_request.email,
        mobile=cv_request.mobile,
        cv_base64=cv_request.pdf_base64,
        score=result.score,
        feedback=result.feedback,
        skills_matched=result.skills_matched,
        total_skills=result.total_skills,
        total_priority=result.total_priority,
        matched_priority=result.matched_priority,
        passed=result.passed
    )
    try:
        await database.execute(query)
    except sqlalchemy.exc.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")

    return result

def get_current_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = credentials.username == ADMIN_USERNAME
    correct_password = credentials.password == ADMIN_PASSWORD
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/api/users")
async def get_users(admin: str = Depends(get_current_admin)):
    query = users.select().order_by(users.c.score.desc())
    return await database.fetch_all(query)

@app.get("/api/users/{user_id}")
async def get_user(user_id: int, admin: str = Depends(get_current_admin)):
    query = users.select().where(users.c.id == user_id)
    user = await database.fetch_one(query)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/api/passed_users")
async def get_passed_users(admin: str = Depends(get_current_admin)):
    query = users.select().where(users.c.passed == True).order_by(users.c.score.desc())
    return await database.fetch_all(query)

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    frontend_code = '''
import streamlit as st
import requests
import base64
import pandas as pd
import re

API_BASE_URL = "http://localhost:8000/api"

def show_pdf(base64_pdf):
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def authenticate_user(username, password):
    return username == "admin" and password == "admin"

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_number(string):
    pattern = r'^\\d{11}$'
    return bool(re.match(pattern, string))

st.sidebar.title("Navigation 🎈")
choice = st.sidebar.radio("Go to", ["Evaluate CV", "Admin Panel"])

if choice == "Evaluate CV":
    st.title(" ''' + profile['job_title'] + ''' CV Evaluation")

    with st.form(key="cv_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        mobile = st.text_input("Mobile")
        uploaded_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
        submit_button = st.form_submit_button(label="Evaluate CV")

    if submit_button:
        if name == "":
            st.error("Please Enter Your Name")
        elif email == "":
            st.error("Please Enter Your Email")
        elif not validate_email(email):
            st.error("Invalid email format.")
        elif mobile == "" or not validate_number(mobile):
            st.error("Please Enter valid Mobile Number")
        elif uploaded_file is None:
            st.error("Please Upload Your CV")
        else:
            pdf_base64 = base64.b64encode(uploaded_file.read()).decode("utf-8")
            cv_request = {"pdf_base64": pdf_base64, "name": name, "email": email, "mobile": mobile}
            response = requests.post(f"{API_BASE_URL}/evaluate_cv", json=cv_request)
            try:
                result = response.json()
                if response.status_code == 200:
                    st.success("CV evaluated successfully! We will contact you.")
                elif response.status_code == 400:
                    if 'detail' in result and result['detail'] == "Email already registered":
                        st.error("Error: Email already registered")
                    else:
                        st.error("An error occurred while processing the response: " + result.get('detail', 'Unknown error'))
                else:
                    st.error("An unexpected error occurred while processing the response from the server.")
            except Exception as e:
                st.error(f"Error: Email already registered")

elif choice == "Admin Panel":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("Enter your credentials")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.experimental_rerun()
            else:
                st.error("Error: Invalid credentials")
    else:
        st.title("All users 👥")
        response = requests.get(f"{API_BASE_URL}/users", auth=('admin', 'admin'))
        if response.status_code == 200:
            users = response.json()
            user_df = pd.DataFrame(users)
            if 'cv_base64' in user_df.columns:
                user_df = user_df.drop(columns=['cv_base64'])
            if not user_df.empty:
                st.dataframe(user_df, use_container_width=True)
                st.markdown("---")

                selected_function = st.selectbox("Select one", ["Choose something...", "View User by ID", "View Passed Users"])

                if selected_function == "View User by ID":
                    st.markdown("👥")
                    user_id = st.text_input("Enter user ID to view CV")
                    if st.button("View Selected User's CV"):
                        if user_id == "" or not user_id.isdigit():
                            st.error("Please enter a valid number")
                        else:
                            user_response = requests.get(f"{API_BASE_URL}/users/{user_id}", auth=('admin', 'admin'))
                            if user_response.status_code == 200:
                                user_data = user_response.json()
                                user_info = {
                                    "Name": [user_data['name']],
                                    "Email": [user_data['email']],
                                    "Mobile": [user_data['mobile']],
                                    "Score": [user_data['score']],
                                    "Skills Matched": [f"{user_data['skills_matched']}/{user_data['total_skills']}"],
                                    "Total Priority Points": [f"{user_data['matched_priority']}/{user_data['total_priority']}"],
                                    "Passed": ["Yes" if user_data['passed'] else "No"]
                                }
                                user_df = pd.DataFrame(user_info)
                                st.dataframe(user_df, use_container_width=True)
                                st.markdown("---")
                                st.write("Resume:")
                                show_pdf(user_data['cv_base64'])
                            else:
                                st.error("User not found or an error occurred.")

                elif selected_function == "View Passed Users":
                    passed_response = requests.get(f"{API_BASE_URL}/passed_users", auth=('admin', 'admin'))
                    if passed_response.status_code == 200:
                        passed_users = passed_response.json()
                        passed_user_df = pd.DataFrame(passed_users)
                        if 'cv_base64' in passed_user_df.columns:
                            passed_user_df = passed_user_df.drop(columns=['cv_base64'])
                        if not passed_user_df.empty:
                            st.dataframe(passed_user_df, use_container_width=True)
                        else:
                            st.info("No passed users found.")
                    else:
                        st.error("An error occurred while fetching passed users.")

        else:
            st.error("An error occurred while fetching users.")

        st.sidebar.markdown("---")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()
'''
    return backend_code, frontend_code
def customized_analyzer():
    st.markdown("---")
    st.header("Create Job Profile 📋")
    
    job_title = st.text_input("Job Title", key="job_title", value=st.session_state.get('job_title', ''))
    skills = st.text_area("Skills (one per line, format: skill_name,priority)", key="skills", value=st.session_state.get('skills', '')).lower()
    with st.expander("Explanation of Priority", expanded=False):
        st.info(
            """
            The priority of a skill represents its importance for the job. For example, consider a job profile with the following skills:

            - Python, priority 3
            - SQL, priority 2
            - Communication, priority 1

            When evaluating a CV, the system will check for the presence of these skills. Here are a few scenarios:

            - If the CV mentions "Python" and "SQL" but not "Communication":
                - The skills matched are Python (priority 3) and SQL (priority 2).
                - The total priority points possible are 3 + 2 + 1 = 6.
                - The matched priority points are 3 + 2 = 5.
                - The score would be (5 / 6) * 100 ≈ 83.33%.

            - If the CV mentions only "Communication":
                - The skills matched are Communication (priority 1).
                - The total priority points possible are 3 + 2 + 1 = 6.
                - The matched priority points are 1.
                - The score would be (1 / 6) * 100 ≈ 16.67%.

            - If the CV mentions all three skills:
                - The skills matched are Python, SQL, and Communication.
                - The total priority points possible are 6.
                - The matched priority points are 6.
                - The score would be (6 / 6) * 100 = 100%.

            The pass threshold is the minimum score required to consider the CV suitable for the job.
            """
        )
    pass_threshold = st.number_input("Pass Threshold (%)", min_value=0.0, max_value=100.0, step=0.1, key="pass_threshold", value=st.session_state.get('pass_threshold', 0.0))
    if st.button("Generate Code"):
        if not job_title:
            st.error("Job Title cannot be empty.")
            return
        if not skills:
            st.error("Skills cannot be empty.")
            return
        
        try:
            skills_list = [
                {"name": skill.split(",")[0].strip(), "priority": int(skill.split(",")[1].strip())}
                for skill in skills.split("\n") if skill and len(skill.split(",")) == 2
            ]
            if not skills_list:
                raise ValueError("Invalid skills format.")
        except ValueError:
            st.error("Each skill must be in the format: skill_name,priority (e.g., python,3)")
            return

        profile = {
            "job_title": job_title,
            "skills": skills_list,
            "pass_threshold": pass_threshold
        }

        backend_code, frontend_code = generate_code(profile)
        
        st.session_state['backend_code'] = backend_code
        st.session_state['frontend_code'] = frontend_code

    if 'backend_code' in st.session_state and 'frontend_code' in st.session_state:
        st.markdown("---")
        st.subheader("Backend Code 🔻")
        with st.expander("Expand to see code", expanded=False):
            st.code(st.session_state['backend_code'], language="python")
        st.download_button("Download", data=st.session_state['backend_code'], file_name="backend.py", mime="text/plain")
        st.markdown("---")

        st.subheader("Frontend Code 🔻")
        with st.expander("Expand to see code", expanded=False):
            st.code(st.session_state['frontend_code'], language="python")
        st.download_button("Download", data=st.session_state['frontend_code'], file_name="frontend.py", mime="text/plain")
        st.markdown("---")

        with st.expander("How to Use ❔"):
            st.markdown("""
                ### Prerequisites
                1. **Python Environment**: Ensure you have Python installed on your system.
                2. **Dependencies**: You will need to install the required libraries. Create a `requirements.txt` file with the following content:
                    ```
                    streamlit
                    fastapi
                    pydantic
                    uvicorn
                    pdfminer.six
                    requests
                    ```

                ### Setting Up the Project
                1. **Create the Streamlit App**: Save the provided code into a file named `app.py`.
                2. **Run the Streamlit App**: Launch the Streamlit application using the command:
                    ```
                    streamlit run app.py
                    ```

                ### Using the Streamlit Application
                1. **First Page - Company Portal**:
                    - **Company Name**: Enter your company name.
                    - **Email**: Enter your email address.
                    - **Mobile**: Enter your mobile number.
                    - Click "Submit" to proceed.
                2. **Second Page - Choose an Option**:
                    - **Custom analyzer**: Choose this option to create a custom job profile analyzer.
                    - **Job advertisement**: Choose this option to create a job advertisement.
                3. **Custom analyzer**:
                    - **Create Job Profile**:
                        - **Job Title**: Enter the job title.
                        - **Skills**: Enter the skills in the format `skill_name,priority` (one per line).
                        - **Pass Threshold**: Enter the pass threshold percentage.
                    - **Generate Code**: Click the "Generate Code" button to generate backend and frontend code for you job profile evaluation.
                    - **Download Code**: Download the generated backend and frontend code by clicking the respective download buttons.
                3. **Setting Up the Generated code**:
                    - **Backend Code**: Save the generated backend code as `backend.py`. Run the FastAPI server using:
                        ```
                        python backend.py
                        ```
                    - **Frontend Code**: Save the generated frontend code as `frontend.py`. Launch the frontend Streamlit application using:
                        ```
                        streamlit run frontend.py
                        ```
                ### Evaluating CVs
                1. **Upload CV**: On the frontend Streamlit app, upload a CV in PDF format.
                2. **Evaluate CV**: Click the "Evaluate CV" button to evaluate the uploaded CV against the job profile.
                3. **View Results**: The results, including the score, feedback, skills matched, and pass status, will be displayed on the frontend.
            """)

def generate_advertisement():
    st.title("Job Advertisement Generator 📝")

    def langchain(company_name, job_title, skills, description, salary, location, email):
        if OpenAI is None:
            st.error("Langchain OpenAI module is not available. Please ensure it is installed.")
            return

        try:
            openai_api_key = st.secrets["openai_paid"]["api_key"]
            llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key, max_tokens=700) 
            
            prompt = (f"Create a job advertisement for the following company:\n\n"
                      f"Company Name: {company_name}\n"
                      f"Job Title: {job_title}\n"
                      f"Required Skills: {skills}\n"
                      f"Job Description: {description}\n"
                      f"Salary: {salary}\n per month"
                      f"Location: {location}\n\n"
                      f"Email address: {email}\n"
                      "Please generate a professional and attractive job advertisement.")
            
            response = llm(prompt)
            st.session_state['generated_advertisement'] = response
            st.info(response)
        except Exception as e:
            st.info(f"An error occurred: {e}")

    with st.form("job_form"):
        company_name = st.session_state.get('company_name', '')
        email = st.session_state.get('email', '') 
        job_title = st.text_input("Job Title")
        skills = st.text_area("Required Skills (comma-separated)")
        description = st.text_area("Job Description")
        salary = st.text_input("Salary (per month)")
        location = st.text_input("Location")
        
        submitted = st.form_submit_button("Generate Advertisement")
        
        if submitted:
            if not job_title:
                st.error("Job Title cannot be empty.")
                return
            if not skills:
                st.error("Required Skills cannot be empty.")
                return
            if not description:
                st.error("Job Description cannot be empty.")
                return
            if not salary:
                st.error("Salary cannot be empty.")
                return
            if not location:
                st.error("Location cannot be empty.")
                return
            if not email:
                st.error("Email address cannot be empty.")
                return
            st.session_state['job_title'] = job_title
            langchain(company_name, job_title, skills, description, salary, location, email)

    if 'generated_advertisement' in st.session_state:
        if st.button("Publish Advertisement"):
            job_title = st.session_state.get('job_title', '')
            generated_ad = st.session_state['generated_advertisement']
            try:
                save_advertisement(company_name, job_title, generated_ad, published=True)
                st.success("Advertisement published successfully!")
                del st.session_state['generated_advertisement']
            except AdvertisementError as e:
                st.info(str(e))
            except Exception as e:
                st.error(e)
    

def show_company_page(company_name, email, mobile):
    st.sidebar.markdown("# Choose something ...")
    option = st.sidebar.selectbox("", ["Custom analyzer", "Job advertisement", "Wall"])

    if option == "Custom analyzer":
        st.title(f"Welcome to {company_name} Custom analyzer page !")
        st.subheader("On this page you can customize your own analyzer ⚡")
        customized_analyzer()
    elif option == "Job advertisement":
        generate_advertisement()
    elif option == "Wall":
        st.title("Published Advertisements")
        ads = get_advertisements(published=True)
        current_time = datetime.now()
        for ad in ads:
            expire_time = ad['expire_time']
            if expire_time > current_time:
                st.info(f"### {ad['job_title']}\n\n{ad['advertisement']}\n\n---")

    if st.sidebar.button("Return to first page"):
        st.session_state.submitted = False
        st.experimental_rerun()

    if st.sidebar.button("Reset"):
        for key in ['job_title', 'skills', 'pass_threshold', 'backend_code', 'frontend_code']:
            if key in st.session_state:     
                del st.session_state[key]
        st.experimental_rerun()

if __name__ == "__main__":
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    if not st.session_state.submitted:
        st.title("Company API Portal")
        company_name = st.text_input("Company Name")
        email = st.text_input("Email")
        mobile = st.text_input("Mobile")

        if st.button("Submit"):
            if company_name and email and mobile:
                st.session_state.submitted = True
                st.session_state.company_name = company_name
                st.session_state.email = email
                st.session_state.mobile = mobile
                st.experimental_rerun()
    else:
        company_name = st.session_state.get('company_name', '')
        email = st.session_state.get('email', '')
        mobile = st.session_state.get('mobile', '')
        show_company_page(company_name, email, mobile)