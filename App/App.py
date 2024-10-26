import streamlit as st 
import hydralit_components as hc
import pandas as pd
import re
import base64, random
import time,datetime
import pymysql
import os
import numpy as np
import random
import datetime
import json
import socket
import shutil
import joblib
import platform
import socket
import secrets
import io,random
from datetime import datetime
from openai import OpenAI 
openai_api_key =st.secrets["openai"]["api_key"]
client = OpenAI(api_key=openai_api_key)
client = OpenAI(api_key=st.secrets["openai"]["api_key"])
from openai import OpenAIError, AuthenticationError, RateLimitError  
import plotly.express as px 
from langchain.llms import OpenAI as LangchainOpenAI
import plotly.graph_objects as go
from serpapi import GoogleSearch
import plotly.figure_factory as ff
from streamlit_pills import pills
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from streamlit_custom_notification_box import custom_notification_box
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
import plotly.graph_objects as go
from pdfminer.high_level import extract_text
from pdfminer3.converter import TextConverter
from pdfminer.high_level import extract_text
from streamlit_tags import st_tags
from PIL import Image
from Links import *
from skill_description import *
from company import *
# import nltk
# nltk.download('stopwords')

from llama_index.llms.openai import OpenAI
try:
    from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
    from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader



def show_wall():
    st.markdown("---")
    ads = get_advertisements(published=True)
    st.markdown(f" Total ads: {len(ads)} ")
    current_time = datetime.now()

    for ad in ads:
        expire_time = ad['expire_time']
        if expire_time > current_time:
            with st.expander(f"## {ad['job_title']}🚨 "):
                st.info(f"### {ad['job_title']} 💼\n\n{ad['advertisement']}\n\n---")
        else:
            st.info(f"{ad['job_title']} has expired and will not be shown.")


def get_csv_download_link(df,filename,text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()      
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                    caching=True,
                                    check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    converter.close()
    fake_file_handle.close()
    return text

def remove_cache():
    try:
        if os.path.exists("cache"):
            shutil.rmtree("cache")

    except Exception as e:
        st.error(f"Error occurred while removing cache: {e}")

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)



if 'shuffled_courses' not in st.session_state:
    st.session_state.shuffled_courses = []

if 'no_of_reco' not in st.session_state:
    st.session_state.no_of_reco = 5

def course_recommender(course_list):

    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, st.session_state.no_of_reco)
    st.session_state.no_of_reco = no_of_reco

    if not st.session_state.shuffled_courses:
        st.session_state.shuffled_courses = course_list.copy()
        random.shuffle(st.session_state.shuffled_courses)

    rec_course = []
    for idx, (c_name, c_link) in enumerate(st.session_state.shuffled_courses[:no_of_reco], 1):
        st.markdown(f"({idx}) [{c_name}]({c_link})")
        rec_course.append(c_name)

    return rec_course

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True
    else:
        return False


def validate_number(string):
    pattern = r'^\d{11}$'
    return bool(re.match(pattern, string))


def shuffle_jobs():

    job_lists = [cyber_jobs, network_jobs, ds_jobs, web_jobs, android_jobs, ios_jobs, uiux_jobs]
    for job_list in job_lists:
        random.shuffle(job_list)


def shuffle_videos():

    videos_lists = [resume_videos,interview_videos]
    for videos_lists in videos_lists:
        random.shuffle(videos_lists)

def delete_advertisement(ad_id):
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM advertisements WHERE id = %s", (ad_id,))
        count = cursor.fetchone()[0]
        if count == 0:
            raise Exception(f"Advertisement with ID {ad_id} does not exist.")

        cursor.execute("DELETE FROM advertisements WHERE id = %s", (ad_id,))
        conn.commit()
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        raise Exception("An error occurred while deleting the advertisement.")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def generate_resume(resume_data):

    def generate_ats_resume(resume_data):
        prompt = f"""
        I have the following resume data: {resume_data}.
        Can you create a professional and ATS-compliant resume for me?
        I want a perfect structure, and you can add advice to mention that you can add something here.
        Reply with just the resume, not any other data.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional resume writer."},
                    {"role": "user", "content": prompt}
                ]
            )
            resume_text = response.choices[0].message.content.strip()
            return resume_text

        except OpenAIError as e:
            st.error(f"OpenAI API Error: {e}")
            return None
        
    def generate_html_ats_resume(resume_data):
        prompt = f"""
        I have the following resume data: {resume_data}.
        Can you create a professional and ATS-compliant resume for me?
        I want a perfect structure, and you can add advice to mention that you can add something here.
        Reply with just the resume, not any other data.
        Note: return just as HTML ATS view response, only HTML.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional resume writer."},
                    {"role": "user", "content": prompt}
                ]
            )
            resume_text = response.choices[0].message.content.strip()
            return resume_text

        except OpenAIError as e:
            st.error(f"OpenAI API Error: {e}")
            return None

    col1, _, _, col4 = st.columns([4, 2, 2, 1])

    with col1:
        st.subheader("**ATS Resume Structure**")

    with col4:
        if st.button('❔', help="An ATS-friendly resume is designed to be easily parsed by Applicant Tracking Systems, which are commonly used by employers to screen resumes before they reach a human recruiter. \"Show button\" will display it in a basic structure > **clean layout, standard fonts, and include relevant keywords from the job description.**"):
            st.markdown("")

    st.markdown(" ")
    with st.expander("RESUME TEMPLATE 📑"):
        st.info(""" 
                
        open it from [here](https://docs.google.com/document/d/1pkXYi3Sy0B6lARne6FOS5aUZMI769h3bHx0jGE_8XBU/edit) 🎈

    Tips⚡  

    1. **Use a Clean Layout:** Avoid complex formatting, graphics, or tables.
    2. **Include Keywords:** Match your resume keywords to the job description.
    3. **Use Standard Section Headings:** Common headers like "Work Experience," "Education," and "Skills."
    4. **Use a Standard Font:** Fonts like Arial, Times New Roman, or Calibri.
    5. **Avoid Headers and Footers:** Important information should be in the main body.
    6. **Use Bullet Points:** For easy readability and to highlight key information.
    7. **Submit in the Correct Format:** Preferably as a .docx or .pdf file.
    8. **Provide Detailed Job Titles:** Use full job titles rather than acronyms.
    9. **Include Dates for Each Job:** List the month and year for start and end dates.
    10. **Spell Check and Proofread:** Ensure there are no spelling or grammatical errors.

    These tips will help your resume get past ATS and into the hands of a human recruiter.
                    
                """ )
    st.markdown("")
    st.subheader("Create your own (beta 🚨)")
    tab1 , tab2 = st.tabs(["Txt ✨","HTML 🎈"])

    with tab1:
        
        if st.button("Show with My own data 🧾"):
            ats_resume = generate_ats_resume(resume_data)
            if ats_resume:
                st.subheader("Generated ATS Resume")
                st.text_area("Your ATS Resume structure", ats_resume, height=300)
                st.download_button(label="Download ⬇️", data=ats_resume, file_name="ATS_Resume.txt", mime="text/plain")

    with tab2:
        if st.button("Show with My own data 📑"):
            ats_resume = generate_html_ats_resume(resume_data)
            if ats_resume:
                st.subheader("Generated ATS Resume")
                st.text_area("Your ATS Resume structure", ats_resume, height=300)
                st.download_button(label="Download ⬇️", data=ats_resume, file_name="ATS_Resume.html", mime="text/html")
                st.markdown(ats_resume, unsafe_allow_html=True)

def is_connected():
    try:
        # Google's public DNS server
        socket.create_connection(("8.8.8.8", 53))
        return True
    except OSError:
        return False
    
def get_mocked_location():
    return [30.0444, 31.2357]


def llm(reco_field):

    # MODEL_NAME = "gpt-3.5-turbo"

    if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
        st.error("OpenAI API key not found. Please add it to the Streamlit secrets.")
        return

    openai_api_key = st.secrets["openai"]["api_key"]
    st.subheader("Chat with AI 🦖")

    @st.cache_resource(show_spinner=False)
    def load_data():
        with st.spinner(text="Loading and indexing career resources – hang tight! This should take 1-2 minutes."):
            reader = SimpleDirectoryReader(input_dir="./career_data/", recursive=True)
            docs = reader.load_data()
            service_context = ServiceContext.from_defaults(
                llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5,
                           system_prompt="You are an expert career advisor specializing in resume analysis, CV writing, job interviews, and job searching. Provide accurate, detailed, and helpful advice. Do not hallucinate features.")
            )
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            return index

    def create_career_data():
        if not os.path.exists("career_data"):
            os.makedirs("career_data")
            with open("career_data/placeholder.txt", "w") as file:
                file.write("This is a placeholder text file.")

    def initialize_chat_engine(index):
        if "chat_engine" not in st.session_state or st.session_state.chat_engine is None:
            st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    def llama_chat():
        st.title("Llama Index 🦙")

        create_career_data()

        index = None
        try:
            index = load_data()
        except ValueError as ve:
            st.error(f"Could not load data for llama: {ve}. Please ensure that your data is properly configured.")
            return

        if "llama_messages" not in st.session_state:
            st.session_state.llama_messages = []

        initialize_chat_engine(index)

        try:
            if prompt := st.chat_input("Your question"):
                st.session_state.llama_messages.append({"role": "user", "content": prompt})

            for message in st.session_state.llama_messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if st.session_state.llama_messages and st.session_state.llama_messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chat_engine.chat(st.session_state.llama_messages[-1]["content"])
                        st.write(response.response)
                        message = {"role": "assistant", "content": response.response}
                        st.session_state.llama_messages.append(message)

        except ValueError as ve:
            st.error(f"Could not load OpenAI embedding model: {ve}. Please ensure that your OPENAI_API_KEY environment variable is properly configured. API keys can be found or created at https://platform.openai.com/account/api-keys. Consider using embed_model='local'. Visit our documentation for more embedding options: https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#modules")

        except OpenAIError as e:
            st.error(f"An error occurred: {str(e)}")

    def langchain_quickstart():
        st.title("Langchain 🦜🔗")

        def generate_response(input_text):
            try:
                llm = LangchainOpenAI(temperature=0.7, openai_api_key=openai_api_key, max_tokens=700)
                response = llm(input_text)
                st.info(response)
            except Exception as e:
                st.info(f"An error occurred: {e}")

        with st.form("my_form"):
            text = st.text_area("Enter text:", f"What are the top 3 interview questions in {reco_field} with sample answers?")
            submitted = st.form_submit_button("Submit")
            if submitted:
                generate_response(text)

    def reset():
        st.markdown("---")
        st.sidebar.markdown("# Clear Chat History")
        if st.sidebar.button('🗑️'):
            st.session_state.chat_engine = None
            st.session_state.llama_messages = []

        st.sidebar.markdown("# Clear Queue")
        if st.sidebar.button('🔃' , help ="If your question is not related to the previous one, click here"):
            st.session_state.chat_engine = None


    option = st.selectbox("Select App:", ["Choose something", "Langchain 🦜", "Llama 🦙"])

    if option == "Langchain 🦜":
        langchain_quickstart()
    elif option == "Llama 🦙":
        llama_chat()
        reset()
    else:
        st.info("""
        Here's a short definition for each of our two options:

        1. **Langchain 🦜**: Langchain leverages AI language models like GPT-3.5 for tasks such as text generation, summarization, and question answering.

        2. **LlamaIndex 🦙**: LlamaIndex is a powerful tool for creating and managing large-scale indexes for efficient data retrieval. It can read and process data that we provide for more effective information management.

        For more information, check our [Project Handbook](https://drive.google.com/file/d/14MiNxKsdcYZPlNJue7yx6PKwviL2kA7N/view?usp=drive_link).
""")



def extract_word_counts(pdf_file):

    text = extract_text(pdf_file)
    words = text.split()
    word_counts = pd.Series(words).value_counts().reset_index()
    word_counts.columns = ['Word', 'Count']
    
    return word_counts, len(words)

def plot_word_count_bar(total_words, min_words=100, max_words=1500, optimal_min=450, optimal_max=650):

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = total_words,
        title = {'text': "Word Count"},
        gauge = {
            'axis': {'range': [min_words, max_words]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [min_words, optimal_min], 'color': "red"},
                {'range': [optimal_min, optimal_max], 'color': "green"},
                {'range': [optimal_max, max_words], 'color': "orange"}]
        }
    ))
    
    return fig



def categorize_skills(resume_data):
    cyber_keyword = [
        'penetration testing', 'network security', 'ethical hacking', 'vulnerability assessment', 'incident response',
        'security operations', 'threat intelligence', 'security analysis', 'digital forensics', 'malware analysis',
        'vulnerabilities', 'security architecture', 'risk management', 'troubleshooting', 'compliance', 'encryption',
        'identity and access management', 'secure coding', 'security awareness training', 'firewalls',
        'intrusion detection', 'endpoint security', 'data loss prevention', 'cloud security', 'tryhackme',
        'web application security', 'comptia', 'mobile security', 'wireless security', 'secure software development',
        'reports', 'operations', 'red teaming', 'blue teaming', 'soc', 'network defense', 'vpn', 'windows', 'dhcp',
        'siem', 'ctf', 'oscp', 'ceh', 'cissp', 'cism', 'crisc', 'ejpt', 'metasploit', 'netsparker', 'sqlmap',
        'osint', 'protocols', 'linux', 'os', 'go', 'golang', 'phishing attack', 'cryptography', 'htb', 'kali linux',
        'pentesting', 'exploitation', 'cpoop', 'bash scripting', 'source code review', 'penetration',
        'thick client application security', 'api security', 'cloud security', 'network security', 'web application security',
        'owasp', 'bug'
    ]

    network_keyword = [
        'dhcp', 'servers', 'dns', 'troubleshooting', 'active directory', 'investigations', 'cms', 'routing protocols',
        'reports', 'data collection', 'compliance', 'licensing', 'routing', 'drafting', 'mis', 'budgeting',
        'outsourcing', 'networking', 'saas', 'migration', 'cisco', 'debugging', 'customer service', 'network architecture',
        'operations', 'installation', 'lan', 'network engineering', 'switching', 'juniper', 'ccna', 'ccnp', 'ccie',
        'network protocols', 'subnetting', 'vlan', 'stp', 'ospf', 'bgp', 'firewalls', 'vpn', 'load balancing', 'snmp',
        'wireshark', 'network monitoring', 'sdn', 'cloud networking', 'wireless networking'
    ]

    ds_keyword = [
        'neural networks', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'plotly', 'bokeh', 'statsmodels',
        'scipy', 'sql', 'big data', 'data mining', 'data preprocessing', 'feature engineering', 'exploratory data analysis',
        'ensemble learning', 'cross-validation', 'hyperparameter tuning', 'model evaluation', 'unsupervised learning',
        'supervised learning', 'dimensionality reduction', 'python', 'r', 'hadoop', 'hive', 'mrjob', 'big data',
        'bioinformatics', 'deep learning', 'machine learning'
    ]

    web_keyword = [
        'python', 'react', 'django', 'node.js', 'react.js', 'php', 'laravel', 'magento', 'wordpress', 'javascript',
        'angular.js', 'c#', 'asp.net', 'flask', 'web development', 'frontend development', 'backend development',
        'full-stack development', 'responsive design', 'restful apis', 'microservices', 'css', 'html', 'web frameworks',
        'database'
    ]

    android_keyword = [
        'android', 'flutter', 'kotlin', 'xml', 'kivy', 'mobile development', 'dart', 'mobile app development',
        'android studio', 'material design', 'firebase', 'android sdk', 'user interface design'
    ]

    ios_keyword = [
        'ios', 'swift', 'cocoa', 'cocoa touch', 'xcode', 'plist', 'objective-c', 'uikit', 'mvvm', 'dart', 'swiftui',
        'universal binaries', 'testflight'
    ]

    uiux_keyword = [
        'ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes', 'storyframes',
        'adobe photoshop', 'illustrator', 'adobe after effects', 'premier pro', 'adobe indesign', 'wireframe',
        'user research', 'user experience', 'sketch', 'visual', 'wireframing', 'interactive'
    ]

    category_counts = {
        'Cyber Security': 0,
        'Network Engineer': 0,
        'Data Science': 0,
        'Web Development': 0,
        'Android Development': 0,
        'IOS Development': 0,
        'UI-UX Development': 0
    }

    for skill in resume_data['skills']:
        categories_for_skill = []
        if skill.lower() in cyber_keyword:
            categories_for_skill.append('Cyber Security')
        if skill.lower() in network_keyword:
            categories_for_skill.append('Network Engineer')
        if skill.lower() in ds_keyword:
            categories_for_skill.append('Data Science')
        if skill.lower() in web_keyword:
            categories_for_skill.append('Web Development')
        if skill.lower() in android_keyword:
            categories_for_skill.append('Android Development')
        if skill.lower() in ios_keyword:
            categories_for_skill.append('IOS Development')
        if skill.lower() in uiux_keyword:
            categories_for_skill.append('UI-UX Development')

        for category in categories_for_skill:
            category_counts[category] += 1

    max_category = max(category_counts, key=category_counts.get)
    max_count = category_counts[max_category]
    ties = [category for category, count in category_counts.items() if count == max_count and count != 0]

    return max_category, max_count, ties, category_counts


def determine_candidate_level(resume_text, resume_data):
    cand_level = ''

    if any(word in resume_text for word in ['EXPERIENCE', 'WORK EXPERIENCE', 'Experience', 'Work Experience', 
                                            'CAREER HISTORY', 'HISTORY', 'PROFESSIONAL BACKGROUND', 
                                            'EMPLOYMENT HISTORY']):
        cand_level = "Experienced"

    elif any(word in resume_text for word in ['INTERNSHIP', 'INTERNSHIPS', 'Internship', 'Internships', 
                                            'Jr', 'JUNIOR', 'Junior', 'TRAINEE', 'Trainee']):
        cand_level = "Intermediate"

    elif any(word in resume_text for word in ['Fresher', 'fresher', 'fresh', 'ENTRY LEVEL', 'Entry Level']):
        cand_level = "Fresher"

    elif resume_data['no_of_pages'] < 1:
        cand_level = "NA"

    else:
        cand_level = "NA"

    return cand_level


def job_recommender(jobs_list):

    c = 0
    rec_job = []
    no_of_reco = 3
    for c_name, c_link in jobs_list:
        c += 1
        rec_job.append(c_name)
        rec_job.append(c_link)
        if c == no_of_reco:
            break
    return rec_job

def job_message(reco_field):
    st.title("Welcome to our Job Section!")

    st.markdown(f"""

        Hello our in our {reco_field} job profile 👨‍💻❤️ ! May Allah make you the best in this field.

        There are some job recommendations based on your skills 

        **But If you find a job that requires more experience, consider it as an opportunity to learn and grow.
        There are always ways to improve and develop your skills.**

        Additionally, There are some statistics in your career in bellow 📊
        
    """)

def job_statistics(job):
    dataset = "./Datasets/other_jobs.csv"
    df = pd.read_csv(dataset)
    df_filtered = df[df['Job Title'] == f'{job}']

    st.markdown("## Now let's play with statistics 🤹")

    tab1,tab2,tab3,tab4,tab5 = st.tabs(["Histogram" , "Scatter plot" , "Heatmap" , "Bar Chart" , "Job Designations"])
    numerical_columns = df_filtered.select_dtypes(include=np.number).columns.to_list()

    with tab1:
        histogram_feature = st.selectbox(f" Select feature to histogram" , numerical_columns , index=1)
        fig_hist = px.histogram(df_filtered , x = histogram_feature)
        st.plotly_chart(fig_hist)


    with tab2:

        col1,col2,col3 = st.columns(3)

        with col1:
            x_columns = st.selectbox(" select column on x axis: " , numerical_columns  , index=2)
        with col2:
            y_columns = st.selectbox(" select column on y axis: " , numerical_columns , index=2)
        with col3:
            color = st.selectbox(" select column to be color " , df_filtered.columns , index=3)

        fig_scatter = px.scatter(df_filtered , x = x_columns , y = y_columns,color =color )
        st.plotly_chart(fig_scatter)


    with tab3:
        data = pd.DataFrame(np.random.rand(10, 10), columns=[f'Col{i}' for i in range(10)])

        st.markdown("## Heatmap")
        corr_matrix = df_filtered.corr()
        fig_heatmap = px.imshow(corr_matrix)
        st.plotly_chart(fig_heatmap)

    with tab4:
        top_10_experience = df_filtered['Experience'].value_counts().head(10)


        fig_bar_experience = px.bar(top_10_experience, x=top_10_experience.index, y=top_10_experience.values, 
                                    color=top_10_experience.index, text=top_10_experience.values, 
                                    labels={'x': 'Experience', 'y': 'Count', 'text': 'Count'}, 
                                    template='ggplot2', title='<b> Experience Levels')
        st.plotly_chart(fig_bar_experience)


        top_10_salary_range = df_filtered['Salary Range'].value_counts().head(10)
        fig_bar_salary_range = px.bar(top_10_salary_range, x=top_10_salary_range.index, y=top_10_salary_range.values, 
                                    color=top_10_salary_range.index, text=top_10_salary_range.values, 
                                    labels={'x': 'Salary Range', 'y': 'Count', 'text': 'Count'}, 
                                    template='ggplot2', title='<b>Top 10 Salary Ranges')
        st.plotly_chart(fig_bar_salary_range)

        with tab5:
            top15_job_titles = df['Role'].value_counts()[:15]
            fig = px.bar(y=top15_job_titles.values, x=top15_job_titles.index, text=top15_job_titles.values,
                labels={'y': 'Count', 'x': 'Job Designations'}, title=f'<b> Top Job Designations in { job } jobs')
            st.plotly_chart(fig)


    st.markdown("---")    

    st.markdown(" # All Dataset 📈")
    n_rows = st.slider("Choose number of rows to display", min_value=5, max_value=len(df_filtered), step=1)
    columns_to_show = st.multiselect("Select columns to show", df_filtered.columns.to_list(), default=df_filtered.columns.to_list())

    display_option = st.selectbox("Choose display option", ["Regular", "Transpose"])

    if display_option == "Regular":
        st.dataframe(df_filtered[:n_rows][columns_to_show])
    else:
        st.dataframe(df_filtered[:n_rows][columns_to_show].T)




def get_ai_job_recommendations(skills):
    try:
        if not os.path.exists("cache"):
            os.makedirs("cache")
        if "job_recommendations.pkl" in os.listdir("cache"):
            with open("cache/job_recommendations.pkl", "rb") as file:
                cached_recommendations = joblib.load(file)
                return cached_recommendations

        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. I will give you resume data and you analyze it "
                    "to tell me what job it is related to. Please return the output as a JSON "
                    "containing the top 5 related jobs, with the most accurate suggestion listed first. "
                    "Ensure the format is as follows:\n"
                    "{\n"
                    "  \"top_jobs\": [\n"
                    "    {\n"
                    "      \"job_title\": \"Job Title 1\",\n"
                    "      \"similarity_score\": similarity_score_1\n"
                    "    },\n"
                    "    {\n"
                    "      \"job_title\": \"Job Title 2\",\n"
                    "      \"similarity_score\": similarity_score_2\n"
                    "    },\n"
                    "    {\n"
                    "      \"job_title\": \"Job Title 3\",\n"
                    "      \"similarity_score\": similarity_score_3\n"
                    "    },\n"
                    "    {\n"
                    "      \"job_title\": \"Job Title 4\",\n"
                    "      \"similarity_score\": similarity_score_4\n"
                    "    },\n"
                    "    {\n"
                    "      \"job_title\": \"Job Title 5\",\n"
                    "      \"similarity_score\": similarity_score_5\n"
                    "    }\n"
                    "  ]\n"
                    "}"
                )
            },
            {"role": "user", "content": json.dumps(skills)}
        ])


        assistant_response = response.choices[0].message.content
        job_recommendations = json.loads(assistant_response)

        with open("cache/job_recommendations.pkl", "wb") as file:
            joblib.dump(job_recommendations, file)

        return job_recommendations

    except OpenAIError as e:
        st.markdown("---")
        st.header("AI Job Recommendations ⚡")
        st.info(f"Can you please try again later 🦥 >> {e}")
        st.markdown("---")
        return None

def plot_ai_job_recommendations(job_recommendations):
    job_titles = [job["job_title"] for job in job_recommendations["top_jobs"]]
    similarity_scores = [job["similarity_score"] for job in job_recommendations["top_jobs"]]

    sorted_jobs = sorted(zip(similarity_scores, job_titles), reverse=True)
    similarity_scores, job_titles = zip(*sorted_jobs)

    bar_fig = px.bar(
        x=similarity_scores,
        y=job_titles,
        orientation='h',
        labels={'x': 'Similarity Score', 'y': 'Job Title'},
        title='Top 5 Related Jobs'
    )

    pie_fig = px.pie(
        values=similarity_scores,
        names=job_titles,
        title='Job Recommendations Distribution'
    )

    most_related_job = job_titles[0]
    st.markdown("---")
    st.subheader("AI Job Recommendations ⚡")


    tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])

    with tab1:
        st.plotly_chart(bar_fig)

    with tab2:
        st.plotly_chart(pie_fig)

    st.markdown(f"#### The most related job is: <span style='color:#086ccc'> {most_related_job} </span> 📊" , unsafe_allow_html=True)
    st.markdown("---")

def validate_cv_or_resume(data):
    try:
        prompt = (
            f"Is the following data a valid CV or resume?\n{data}\n"
            "---\n"
            "If not, please provide in short response the reasons why it is not valid."
            "---\n"
            "If yes, please provide an advice about the CV."
        )

        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ])


        validation_response = response.choices[0].message.content.strip()

        if "valid" in validation_response.lower() and "not" not in validation_response.lower():
            return {"validation": "yes", "advice": validation_response}
        else:
            return {"validation": "no", "problem": validation_response}

    except OpenAIError as e:
        return json.dumps({'error': str(e)})


def fetch_job_listings(serp_api_key, query, location, lang="en"):
    params = {
        "engine": "google_jobs",
        "q": f"{query} {location}",
        "hl": lang,
        "serp_api_key": serp_api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    jobs_results = results.get("jobs_results", [])

    return jobs_results

def get_job_details(jobs_results):
    job_details = []
    if jobs_results:
        for job in jobs_results:
            title = job.get('title')
            company = job.get('company_name')
            job_link = None
            if 'related_links' in job and job['related_links']:
                for link in job['related_links']:
                    if 'link' in link:
                        job_link = link['link']
                        break
            job_details.append((title, company, job_link))
    return job_details

def job_pills(jobs_list,reco_field):

    st.markdown("---")
    st.header("Let's dive into job recommendations ✨")
    option = st.selectbox("We hope you find the perfect job!🔍", ["Choose something ...","pills 🎈", "Wall 🧾"])
    if option == "Choose something ...":
        st.info("""
                
        Pills 🎈: Customized job suggestions based on your profession and skills.    
                  
        Wall 🧾: A public board displaying all job advertisements posted by companies with us.     
                
                """)
    elif option == "Wall 🧾":
        show_wall()
    else: 

        pill = job_recommender(jobs_list)
        selected = pills("", ["job 1", "job 2", "job 3","job 4", "job 5"], ["☝️", "✌️", "👌" , "✨","✨"])
        if selected == "job 1":
            st.markdown(f"[{pill[0]}]({pill[1]})")

        elif selected == "job 2":
            st.markdown(f"[{pill[2]}]({pill[3]})")  

        elif selected == "job 3":
            st.markdown(f"[{pill[4]}]({pill[5]})")



        elif selected == "job 4":

            try:
                serp_api_key = st.secrets["serpapi"]["api_key"]
                query = f"{reco_field}"
                location = ""

                jobs_results = fetch_job_listings(serp_api_key, query, location)
                job_details = get_job_details(jobs_results)

                if job_details:
                    title, company, job_link = job_details[1]
                    st.markdown(f"[{title} at {company}]({job_link})")
                else:
                    st.write("Sorry, this job is not available for you. Check the other jobs 😉❤️")
            except Exception as e:
                st.info(f"Check internet connection 🚨")


        elif selected == "job 5":

            try:
                serp_api_key = st.secrets["serpapi"]["api_key"]
                query = f"{reco_field}"
                location = ""

                jobs_results = fetch_job_listings(serp_api_key, query, location)
                job_details = get_job_details(jobs_results)

                if job_details:
                    title, company, job_link = job_details[2]
                    st.markdown(f"[{title} at {company}]({job_link})")
                else:
                    st.write("Sorry, this job is not available for you. Check the other jobs 😉❤️")
            except Exception as e:
                st.info(f"Check internet connection 🚨")



connection = pymysql.connect(host='localhost',user='root',password='',db='cv')
cursor = connection.cursor()


def insert_data(sec_token, ip_add, host_name, dev_user, os_name_ver, latlong, city, state, country, act_name, act_mail, act_mob, name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses, pdf_name):
    DB_table_name = 'user_data'

    insert_sql = """
        INSERT INTO {table_name} 
        (sec_token, ip_add, host_name, dev_user, os_name_ver, latlong, city, state, country, act_name, act_mail, act_mob, Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, Actual_skills, Recommended_skills, Recommended_courses, pdf_name) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
        ON DUPLICATE KEY UPDATE 
        sec_token=VALUES(sec_token), 
        ip_add=VALUES(ip_add), 
        host_name=VALUES(host_name), 
        dev_user=VALUES(dev_user), 
        os_name_ver=VALUES(os_name_ver), 
        latlong=VALUES(latlong), 
        city=VALUES(city), 
        state=VALUES(state), 
        country=VALUES(country), 
        act_name=VALUES(act_name), 
        act_mail=VALUES(act_mail), 
        act_mob=VALUES(act_mob), 
        Name=VALUES(Name), 
        Email_ID=VALUES(Email_ID), 
        resume_score=VALUES(resume_score), 
        Timestamp=VALUES(Timestamp), 
        Page_no=VALUES(Page_no), 
        Predicted_Field=VALUES(Predicted_Field), 
        User_level=VALUES(User_level), 
        Actual_skills=VALUES(Actual_skills), 
        Recommended_skills=VALUES(Recommended_skills), 
        Recommended_courses=VALUES(Recommended_courses)
    """.format(table_name=DB_table_name)

    rec_values = (str(sec_token), ip_add, host_name, dev_user, os_name_ver, str(latlong), city, state, country, act_name, act_mail, act_mob, name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, skills, recommended_skills, courses, pdf_name)

    cursor.execute(insert_sql, rec_values)
    connection.commit()

def insertf_data(feed_name,feed_email,feed_score,comments,Timestamp):
    DBf_table_name = 'user_feedback'
    insertfeed_sql = "insert into " + DBf_table_name + """
    values (0,%s,%s,%s,%s,%s)"""
    rec_values = (feed_name, feed_email, feed_score, comments, Timestamp)
    cursor.execute(insertfeed_sql, rec_values)
    connection.commit()

st.set_page_config(
   page_title="AI Resume Analyzer",
   page_icon='💼',
)


def run():



    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    if "user_type" not in st.session_state:
        st.session_state.user_type = None

    if st.session_state.user_type is None:

        img = Image.open('./Photos/Cyber_Cloud.png')
        st.image(img)

        st.title("Are you an individual or a company?")
        user_type = st.selectbox("Choose one", ["", "Individual", "Company"])

        with st.expander("Cyber Cloud 🌐☁️"):
            
            st.info("""
    **Cyber Cloud: Practice Today, Succeed Tomorrow** 🌐☁️

    Welcome to Cyber Cloud! We are a fictional company designed to help you practice and improve your job application skills. Our tools let you practice without risking your actual job applications.

    **Our Mission:**
    At Cyber Cloud, we help you get better at job hunting by providing:

    1. **CV Analysis Simulation:** 📄💼
    - **For Job Seekers:** Upload your CV and get instant feedback on how to make it better. Fix common issues and improve your chances of getting noticed by real companies.
    - **For Employers:** Simulate the process of screening CVs to identify the best candidates quickly and fairly.

    2. **Mock Interview Simulation:** 🗣️👥
    - **For Job Seekers:** Practice real interview questions and get feedback on your answers. Improve your interview skills and build confidence.
    - **For Employers:** Use our interactive bot to simulate interview scenarios and improve your interviewing techniques.

    **Why Choose Cyber Cloud?**
    - **Tailored Feedback:** Get personalized tips for your CV and interview skills.
    - **Confidence Building:** Practice in a safe environment before facing real interviews.
    - **Streamlined Hiring:** Employers can refine their recruitment process efficiently.

    **Our Vision:**
    To help job seekers and employers practice and improve their skills, making the job application process smoother and more effective.

    Remember, Cyber Cloud is a practice environment. Test your skills here before applying to your real job! ✨🚀
            """)
        if user_type:
            st.session_state.user_type = user_type
            st.experimental_rerun()

    if not st.session_state.submitted:
        if st.session_state.user_type == "Individual":
            img = Image.open('./Photos/Cyber.png')
            st.image(img)
            st.title("Start your journey with us ✨")

            st.session_state.shuffled_courses = []
            remove_cache()

            name = st.text_input("Enter your name")
            email = st.text_input("Enter your email")
            mobile = st.text_input("Enter your mobile number")

            pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])

            if pdf_file is not None:
                # try:
                    save_image_path = './Uploaded_Resumes/' + pdf_file.name
                    pdf_name = pdf_file.name
                    with open(save_image_path, "wb") as f:
                        f.write(pdf_file.getbuffer())


                    # extracted_text = extract_text(save_image_path)
                    # result = validate_cv_or_resume(extracted_text)

                #     if isinstance(result, str):
                #             try:
                #                 validation_result = json.loads(result)
                #             except json.JSONDecodeError:
                #                 st.error("Failed to decode JSON response")
                #                 validation_result = None
                #     else:
                #             validation_result = result

                #     if validation_result is not None:
                        
                #             validation = validation_result.get('validation', '')
                #             problem = validation_result.get('problem', '')
                #             advice = validation_result.get('advice', '')
                    
                #             if "yes" in validation.lower():
                #                 with st.expander("**Valid resume ✅**"):
                #                     st.write(advice)
                #                     # st.write(advice.split('.', 1)[1].strip())  
                #             elif OpenAIError:
                #                 st.info(f"Check internet connection or your api key ☹️")
                #             else:
                #                 with st.expander("**Invalid resume ❌**"):
                #                     st.write(problem ) 

                # except OpenAIError as e:
                #     st.write(f"OpenAI API Error: {e}")
                # except Exception as e:
                #         st.write(f"An error occurred: {e}")

            col1, _, _, _, _, _, col7 = st.columns(7)

            with col1:
                submit_button = st.button("Submit")

            if submit_button:
                if name == "":
                    st.error("Please Enter Your Name")
                elif email == "":
                    st.error("Please Enter Your Email")
                elif not validate_email(email):
                    st.error("Invalid email format.")
                elif mobile == "" or not validate_number(mobile):
                    st.error("Please Enter valid Mobile Number")
                elif pdf_file is None:
                    st.error("Please Enter Resume")
                else:
                    with st.spinner('Hang On While We Cook Magic For You  ...✨'):
                        time.sleep(5)

                    resume_data = ResumeParser(save_image_path).get_extracted_data()

                    st.session_state.name = name
                    st.session_state.email = email
                    st.session_state.mobile = mobile
                    st.session_state.pdf_file = pdf_name
                    st.session_state.pdf_path = save_image_path
                    st.session_state.resume_data = resume_data
                    st.session_state.submitted = True

                    st.experimental_rerun()

            with col7:
                if st.button("🏠" , help="Back to home page "):
                    st.session_state.user_type = None
                    st.experimental_rerun()


        elif st.session_state.user_type == "Company":
            st.title("Register your company with us ✨")
            if 'generated_advertisement' in st.session_state:
                del st.session_state['generated_advertisement']
            remove_cache()

            company_name = st.text_input("Enter your company name")
            email = st.text_input("Enter company email")
            mobile = st.text_input("Enter company mobile number")

            col1,_,_,_,_,_,col7 = st.columns(7)

            with col1:
                submit_button = st.button("Submit")

            if submit_button:
                if company_name == "":
                    st.error("Please Enter Your Company Name")
                elif email == "":
                    st.error("Please Enter Your Email")
                elif not validate_email(email):
                    st.error("Invalid email format.")
                elif mobile == "":
                    st.error("Please Enter valid Mobile Number")
                else:
                    st.session_state.company_name = company_name
                    st.session_state.email = email
                    st.session_state.mobile = mobile
                    st.session_state.submitted = True

                    st.experimental_rerun()

            with col7:

                if st.button("🏠" , help = "Back to home page "):
                    st.session_state.user_type = None
                    st.experimental_rerun()

    else:
        if st.session_state.user_type == "Individual":
            show_individuals_page(st.session_state.name, st.session_state.email, st.session_state.mobile, st.session_state.pdf_path, st.session_state.resume_data, st.session_state.pdf_file)
        elif st.session_state.user_type == "Company":
            show_company_page(st.session_state.company_name, st.session_state.email, st.session_state.mobile)


    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)

    DB_table_name = 'user_data'
    table_sql = """
        CREATE TABLE IF NOT EXISTS {table_name} (
            ID INT NOT NULL AUTO_INCREMENT,
            sec_token VARCHAR(20) NOT NULL,
            ip_add VARCHAR(50) NULL,
            host_name VARCHAR(50) NULL,
            dev_user VARCHAR(50) NULL,
            os_name_ver VARCHAR(50) NULL,
            latlong VARCHAR(50) NULL,
            city VARCHAR(50) NULL,
            state VARCHAR(50) NULL,
            country VARCHAR(50) NULL,
            act_name VARCHAR(50) NOT NULL,
            act_mail VARCHAR(50) NOT NULL,
            act_mob VARCHAR(20) NOT NULL,
            Name VARCHAR(500) NOT NULL,
            Email_ID VARCHAR(500) NOT NULL,
            resume_score VARCHAR(8) NOT NULL,
            Timestamp VARCHAR(50) NOT NULL,
            Page_no VARCHAR(5) NOT NULL,
            Predicted_Field BLOB NOT NULL,
            User_level BLOB NOT NULL,
            Actual_skills BLOB NOT NULL,
            Recommended_skills BLOB NOT NULL,
            Recommended_courses BLOB NOT NULL,
            pdf_name VARCHAR(50) NOT NULL,
            PRIMARY KEY (ID),
            UNIQUE KEY (Email_ID)
        )
    """.format(table_name=DB_table_name)

    cursor.execute(table_sql)


    DBf_table_name = 'user_feedback'
    tablef_sql = "CREATE TABLE IF NOT EXISTS " + DBf_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                        feed_name varchar(50) NOT NULL,
                        feed_email VARCHAR(50) NOT NULL,
                        feed_score VARCHAR(5) NOT NULL,
                        comments VARCHAR(100) NULL,
                        Timestamp VARCHAR(50) NOT NULL,
                        PRIMARY KEY (ID)
                    );
                """
    cursor.execute(tablef_sql)


def show_individuals_page(name, email, mobile, pdf_path, resume_data,pdf_name):

    st.sidebar.markdown("# Choose Something")
    activities = ["User", "Jobs", "Feedback", "Ai" ,"Admin"]
    choice = st.sidebar.selectbox("", activities)

    st.sidebar.markdown("# Upload another Resume")

    if st.sidebar.button("Return to first page"):
        st.session_state.submitted = False
        st.experimental_rerun()


    if choice == 'User':

        shuffle_jobs()
        shuffle_videos()
        act_name = name
        act_mail = email
        act_mob  = mobile
        sec_token = secrets.token_urlsafe(12)
        host_name = socket.gethostname()
        ip_add = socket.gethostbyname(host_name)
        dev_user = os.getlogin()
        os_name_ver = platform.system() + " " + platform.release()

        if is_connected():
            import geocoder
            from geopy.geocoders import Nominatim
            g = geocoder.ip('me')
            latlong = g.latlng
            geolocator = Nominatim(user_agent="http")
            location = geolocator.reverse(latlong, language='en')
            address = location.raw['address']
            city = address.get('city', '')
            state = address.get('state', '')
            country = address.get('country', '')
        else:
            latlong = get_mocked_location()
            city = "Cairo"
            state = "Cairo Governorate"
            country = "Egypt"


        if resume_data:
            resume_text = pdf_reader(pdf_path)
            st.title("**Your Basic info 👀**")
            with st.expander("**Note✨: If you find any incorrect information, you may have written your CV incorrectly. Send us the problem in the feedback section, and we will try to help you**"):
               
                try:
                    
                    name = resume_data.get('name')
                    email = resume_data.get('email')
                    contact = resume_data.get('mobile_number')
                    degree = resume_data.get('degree')
                    no_of_pages = resume_data.get('no_of_pages')
                    data = f"I see that your name is {name} and your mail is {email} and your degree is {degree}"

                    st.markdown("""
                        <style>
                        .tooltip {
                            position: relative;
                            display: inline-block;
                            cursor: not-allowed;
                        }

                        .tooltip .tooltiptext {
                            visibility: hidden;
                            width: 220px;
                            background-color: #555;
                            color: #fff;
                            text-align: center;
                            padding: 5px;
                            position: absolute;
                            z-index: 1;
                            bottom: 125%; /* Position the tooltip above the button */
                            left: 50%;
                            margin-left: -110px; /* Use margin to center the tooltip */
                            opacity: 0;
                            transition: opacity 0.3s;
                        }

                        .tooltip:hover .tooltiptext {
                            visibility: visible;
                            opacity: 1;
                        }

                        .disabled-button {
                            pointer-events: none;
                            opacity: 0.65;
                            cursor: not-allowed;
                            border: none; 
                            background: none; 
                            font-size: 16px; 
                            color: #007BFF; 
                        }
                        </style>
                        """, unsafe_allow_html=True)

                    button_html = f'''
                        <div class="tooltip">
                            <button class="disabled-button" disabled> ✨ </button>
                            <span class="tooltiptext">{data}</span>
                        </div>
                    '''


                    st.text('Name: ' + ('✅ ' + "" if name else '❌ Missing'))
                    st.text('Email: ' + ('✅ ' + "" if email else '❌ Missing'))
                    st.text('Contact: ' + ('✅ ' + "" if contact else '❌ Missing'))
                    st.text('Degree: ' + ('✅ ' + "" if degree and str(degree) != "None" else '❌ Missing'))
                    st.text('Resume pages: ' + (str(no_of_pages) if no_of_pages else '❌ Missing'))
                    st.markdown(button_html, unsafe_allow_html=True)
                except:
                    pass

            st.markdown("---")
            st.subheader("**Your Resume 📑**")
            show_pdf(pdf_path)


            cand_level = determine_candidate_level(resume_text, resume_data)
            if cand_level == "Experienced":
                st.markdown('---', unsafe_allow_html=True)
                st.markdown("<h4 style='text-align: left; color: #611096;'>You are at the experienced level! 😎</h4>", unsafe_allow_html=True)
                st.markdown('---', unsafe_allow_html=True)

            elif cand_level == "Intermediate":
                st.markdown('---', unsafe_allow_html=True)
                st.markdown("<h4 style='text-align: left; color: #611096;'>You are at the intermediate level! 👨‍💻</h4>", unsafe_allow_html=True)
                st.markdown('---', unsafe_allow_html=True)

            elif cand_level == "Fresher":
                st.markdown('---', unsafe_allow_html=True)
                st.markdown("<h4 style='text-align: left; color: #611096;'>You are at the fresher level! 🐣</h4>", unsafe_allow_html=True)
                st.markdown('---', unsafe_allow_html=True)

            elif cand_level == "NA":
                st.markdown('---', unsafe_allow_html=True)
                st.markdown("<h4 style='text-align: left; color: #611096;'>Is this a real resume? 🦥</h4>", unsafe_allow_html=True)
                st.markdown('---', unsafe_allow_html=True)

            else:
                st.markdown('---', unsafe_allow_html=True)



            st.title("Resume Word Count Analysis")

            if pdf_path:

                word_counts, total_words = extract_word_counts(pdf_path)
                average_min_words = 450
                average_max_words = 650

                fig = plot_word_count_bar(total_words)
                st.plotly_chart(fig)
                
                if total_words < average_min_words:
                    st.warning(f"Your resume's word count is slightly short ☹️\n\nYour resume has **{total_words}** words. This is slightly below the average word count of top resumes.")

                elif total_words > 1500:
                    st.warning(f"I founded **{total_words}** words. This may be cv not resume.")
                elif total_words > average_max_words:
                    st.warning(f"Your resume's word count is slightly long ☹️\n\nYour resume has **{total_words}** words. This is slightly above the average word count of top resumes.")
                else:
                    st.success(f"Your resume has a good word count 🥳!\n\nYour resume has **{total_words}** words, which is within the average word count of top resumes.")
                
                st.info(f"A concise, targeted resume shows recruiters your ability to synthesize, prioritize, and convey your most important achievements. Top resumes often are between {average_min_words} and {average_max_words} words long. Your resume has {total_words} words.")


                st.markdown("---")

            st.subheader("**Skills Recommendation 💡**")

            keywords = st_tags(label='# Your Current Skills',
                            text='See our skills recommendation below',
                            value=resume_data['skills'],
                            key='1')

            max_category, max_count, ties , category_counts = categorize_skills(resume_data)

            if len(ties) > 1 or max_count == 0:
                reco_field = 'NA'
                st.warning("**Currently, our tool only predicts and recommends for Cyber Security, Data Science, Web Development, Android Development, iOS Development, UI/UX Development, and Network Engineering. If your job is one from them, you must edit your resume because it contains the equal keywords in two jobs.**")
                recommended_skills = ['No Recommendations']
                sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                categories, counts = zip(*sorted_categories)
                fig = go.Figure(data=[go.Bar(x=categories, y=counts)])
                fig.update_layout(title="Final Count of Job Categories", xaxis_title="Job Category", yaxis_title="Count")
                st.plotly_chart(fig)

                recommended_keywords = st_tags(label='# Recommended skills & Advices for you.',
                                            text='Currently No Recommendations',
                                            value=recommended_skills,
                                            key='8')

                st.markdown('''<h5 style='text-align: left; color: #092851;'>Maybe Available in Future Updates</h5>''', unsafe_allow_html=True)
                rec_course = "Sorry! Not Available for this Field"



            else:
                reco_field = max_category
                st.success(f"**Our analysis says you are looking for {reco_field} Jobs**")

                sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                categories, counts = zip(*sorted_categories)
                fig = go.Figure(data=[go.Bar(x=categories, y=counts)])
                fig.update_layout(title="Final Count of Job Categories", xaxis_title="Job Category", yaxis_title="Count")
                st.plotly_chart(fig)


                recommended_skills = []
                recommended_skills_not_in_resume = []
                rec_course = None

                if reco_field == 'Cyber Security':


                    if cand_level == "Fresher" or cand_level == "NA" or cand_level == "Intermediate":
                        recommended_skills = [
                            "Continuous Learning: Stay updated with the latest cybersecurity trends, technologies, and threats.",
                            "Ethical Conduct: Prioritize ethical behavior and adhere to legal and regulatory standards.",
                            "Hands-On Experience: Gain practical experience through internships, projects, and competitions.",
                            "Technical Skills: Develop proficiency in networking, operating systems, programming, and security tools.",
                            "Soft Skills: Cultivate analytical thinking, problem-solving, communication, and teamwork abilities.",
                            "Certifications: Obtain industry-recognized certifications like CompTIA Security+, CISSP, CEH, or CISM.",
                            "Stay Informed: Read blogs, attend conferences, and join professional organizations to stay updated.",
                            "Networking: Build a professional network within the cybersecurity community for mentorship and career opportunities.",
                            "Specialization: Consider specializing in areas such as network security, cloud security, or incident response.",
                        ]
                    elif cand_level == "Experienced" :
                        recommended_skills = [
                            "Advanced Threat Detection: Master advanced techniques for identifying and mitigating sophisticated cyber threats, including zero-day attacks and advanced persistent threats (APTs).",
                            "Incident Response Planning: Develop comprehensive incident response plans and procedures for handling security incidents effectively and minimizing impact.",
                            "Penetration Testing: Acquire expertise in conducting thorough penetration tests to identify vulnerabilities in systems and networks before attackers do.",
                            "Security Automation and Orchestration: Implement automated solutions for security tasks such as threat detection, incident response, and compliance monitoring.",
                            "Cloud Security Architecture: Design and implement secure cloud architectures to protect data and applications hosted in cloud environments like AWS, Azure, or Google Cloud Platform.",
                            "Blockchain Security: Understand the security implications of blockchain technology and develop strategies for securing blockchain-based systems and applications.",
                            "Threat Intelligence Analysis: Analyze threat intelligence feeds and data sources to identify emerging threats and develop proactive security measures.",
                            "Cybersecurity Governance and Compliance: Gain expertise in cybersecurity regulations, standards, and frameworks such as GDPR, HIPAA, ISO 27001, and NIST.",
                            "Advanced Cryptography: Explore advanced cryptographic techniques for data protection, including homomorphic encryption, post-quantum cryptography, and blockchain-based cryptography.",
                        ]

                    recommended_skills_selected = random.sample(recommended_skills, 3)
                    recommended_skills_not_in_resume = [skill for skill in recommended_skills_selected if skill not in resume_data['skills']]
                    rec_course = course_recommender(cyber_course)


                elif reco_field == 'Network Engineer':


                    if cand_level == "Fresher" or cand_level == "NA" or cand_level == "Intermediate": 

                            recommended_skills = [
                                "TCP/IP Networking: Understand the TCP/IP protocol suite and its components such as IP addressing, subnetting, and routing.",
                                "Network Devices Configuration: Configure and manage network devices such as routers, switches, firewalls, and access points.",
                                "Network Protocols: Familiarize yourself with common network protocols such as DHCP, DNS, SNMP, and HTTP/HTTPS.",
                                "Network Security: Learn basic network security principles and techniques such as access control lists (ACLs), VPNs, and firewalls.",
                                "Troubleshooting Skills: Develop troubleshooting skills to identify and resolve network issues using tools like ping, traceroute, and Wireshark.",
                                "Network Monitoring: Set up network monitoring tools and systems to monitor network performance and detect anomalies.",
                                "Virtualization: Gain knowledge of network virtualization concepts and technologies such as VLANs, VRFs, and virtual switches.",
                                "Wireless Networking: Understand wireless networking concepts and standards such as IEEE 802.11 (Wi-Fi) and implement wireless networks.",
                                "Network Documentation: Learn to create and maintain network documentation including network diagrams, configurations, and inventory.",
                            ]
                    elif cand_level =="Experienced" :
                            recommended_skills = [
                                "Advanced Routing and Switching: Deepen your understanding of routing protocols such as OSPF, EIGRP, BGP, and switch technologies like VLANs, STP, and EtherChannel.",
                                "Network Automation: Learn network automation concepts and tools such as Ansible, Python scripting, and APIs for automating network configuration and management tasks.",
                                "Software-Defined Networking (SDN): Explore SDN principles and technologies such as OpenFlow, SD-WAN, and network programmability for flexible and dynamic network management.",
                                "Cloud Networking: Gain expertise in designing and implementing network solutions in cloud environments like AWS, Azure, or Google Cloud Platform.",
                                "Network Security Hardening: Implement advanced security measures such as intrusion detection/prevention systems (IDS/IPS), network segmentation, and DDoS mitigation techniques.",
                                "Advanced Troubleshooting: Develop advanced troubleshooting skills using packet capture analysis, network simulation tools, and protocol analyzers to diagnose complex network issues.",
                                "Network Design and Architecture: Design scalable and resilient network architectures considering factors such as performance, reliability, and security.",
                                "High Availability and Redundancy: Implement high availability and redundancy mechanisms such as load balancing, link aggregation, and failover clustering for critical network services.",
                                "Network Performance Optimization: Optimize network performance through traffic engineering, Quality of Service (QoS) policies, and performance tuning techniques.",
                            ]
                    recommended_skills_selected = random.sample(recommended_skills, 3)
                    recommended_skills_not_in_resume = [skill for skill in recommended_skills_selected if skill not in resume_data['skills']]
                    rec_course = course_recommender(network_course)

                elif reco_field == 'Data Science':



                    if cand_level == "Fresher" or cand_level == "NA" or cand_level == "Intermediate":
                        recommended_skills = [
                            "Statistical Analysis: Master statistical techniques for analyzing data and deriving insights.",
                            "Programming Languages: Learn languages such as Python, R, or SQL for data manipulation and analysis.",
                            "Machine Learning: Gain proficiency in algorithms and techniques for predictive modeling and pattern recognition.",
                            "Data Visualization: Develop skills in tools like Matplotlib, Seaborn, or Tableau for presenting data effectively.",
                            "Big Data Technologies: Familiarize yourself with frameworks like Hadoop, Spark, or Apache Flink for processing large datasets.",
                            "Data Cleaning and Preprocessing: Understand techniques for handling missing data, outlier detection, and feature engineering.",
                            "Domain Knowledge: Acquire expertise in the specific domain you're working in, such as finance, healthcare, or marketing.",
                            "Communication Skills: Effectively communicate findings and insights to stakeholders through reports, presentations, and visualizations.",
                            "Version Control: Use tools like Git for version control and collaboration on data science projects.",
                        ]
                    elif cand_level =="Experienced" :
                        recommended_skills = [
                            "Deep Learning: Master advanced neural network architectures like CNNs, RNNs, and GANs for complex data analysis tasks.",
                            "Natural Language Processing (NLP): Learn techniques for processing and analyzing large volumes of text data, including sentiment analysis, topic modeling, and named entity recognition.",
                            "Distributed Computing: Gain expertise in distributed computing frameworks such as Apache Spark and Dask for processing massive datasets in parallel.",
                            "Reinforcement Learning: Explore advanced techniques for training agents to make sequential decisions in dynamic environments.",
                            "Time Series Analysis: Understand methods for analyzing time-dependent data, including forecasting, anomaly detection, and causal inference.",
                            "Bayesian Methods: Utilize Bayesian statistical techniques for probabilistic modeling, uncertainty quantification, and decision making.",
                            "Advanced Visualization: Enhance data presentation skills with interactive visualizations, dashboards, and storytelling techniques.",
                            "Cloud Computing: Learn to deploy and scale data science solutions in cloud environments using platforms like AWS, Azure, or Google Cloud.",
                            "Model Interpretability and Explainability: Develop techniques for understanding and explaining complex machine learning models to stakeholders.",
                        ]
                    recommended_skills_selected = random.sample(recommended_skills, 3)
                    recommended_skills_not_in_resume = [skill for skill in recommended_skills_selected if skill not in resume_data['skills']]
                    rec_course = course_recommender(ds_course)



                elif reco_field == 'Web Development':



                    if cand_level == "Fresher" or cand_level == "NA" or cand_level == "Intermediate":
                        recommended_skills = [
                            "Responsive Web Design: Master techniques for creating websites that adapt to various screen sizes and devices.",
                            "Frontend Frameworks: Learn popular frontend frameworks like React, Vue.js, or Angular for building interactive user interfaces.",
                            "Backend Development: Develop skills in server-side programming with languages like Node.js, Python (Django/Flask), or Ruby on Rails.",
                            "Database Management: Understand database concepts and learn to work with relational databases (MySQL, PostgreSQL) or NoSQL databases (MongoDB, Firebase).",
                            "Version Control Systems: Use Git for version control to track changes and collaborate with other developers effectively.",
                            "API Integration: Learn to integrate third-party APIs into web applications for additional functionality and data retrieval.",
                            "Web Security: Understand common web security threats and implement security best practices such as input validation, authentication, and authorization.",
                            "Performance Optimization: Optimize website performance by minimizing load times, optimizing images, and caching resources.",
                            "Debugging and Testing: Develop skills in debugging code and writing automated tests to ensure the reliability of web applications.",
                        ]
                    elif cand_level =="Experienced" :
                        recommended_skills = [
                            "Advanced JavaScript: Master advanced JavaScript concepts such as closures, promises, async/await, and functional programming techniques.",
                            "Full-Stack Development: Gain expertise in both frontend and backend development, including frameworks like React, Vue.js, Angular, Node.js, Express, and MongoDB.",
                            "Serverless Architecture: Learn to build scalable web applications using serverless technologies like AWS Lambda, Azure Functions, or Google Cloud Functions.",
                            "Progressive Web Apps (PWAs): Develop PWAs to provide a native app-like experience using web technologies, including service workers, push notifications, and offline support.",
                            "GraphQL: Explore GraphQL as an alternative to RESTful APIs for efficient data fetching and manipulation.",
                            "Containerization and Orchestration: Understand containerization with Docker and container orchestration with Kubernetes for deploying and managing web applications at scale.",
                            "Microservices Architecture: Design and implement web applications using a microservices architecture for improved scalability, resilience, and flexibility.",
                            "WebAssembly: Learn to compile high-level languages like C/C++ or Rust to run in web browsers for high-performance web applications.",
                            "Web Security Best Practices: Deepen your understanding of web security principles and implement advanced security measures such as content security policy (CSP), HTTP security headers, and secure coding practices.",
                        ]
                    recommended_skills_selected = random.sample(recommended_skills, 3)
                    recommended_skills_not_in_resume = [skill for skill in recommended_skills_selected if skill not in resume_data['skills']]
                    rec_course = course_recommender(web_course)



                elif reco_field == 'Android Development':



                    if cand_level == "Fresher" or cand_level == "NA" or cand_level == "Intermediate":
                            recommended_skills = [
                                "Java or Kotlin Programming: Learn Java or Kotlin programming languages for Android app development.",
                                "User Interface (UI) Design: Master UI design principles and create visually appealing and intuitive user interfaces.",
                                "Android SDK: Familiarize yourself with the Android Software Development Kit (SDK) and its components like activities, fragments, intents, and layouts.",
                                "Database Management: Understand SQLite databases and learn to implement local data storage and retrieval in Android apps.",
                                "Networking: Gain proficiency in making network requests and handling responses using libraries like Retrofit or Volley.",
                                "User Authentication: Implement user authentication and authorization features using techniques like OAuth, Firebase Authentication, or JWT.",
                                "Version Control: Use Git for version control to track changes and collaborate with other developers effectively.",
                                "Testing and Debugging: Develop skills in testing Android apps using frameworks like JUnit, Espresso, and Mockito, and debugging using Android Studio tools.",
                                "Material Design: Implement Material Design guidelines to create consistent and visually appealing Android apps.",
                            ]
                    elif cand_level =="Experienced" :
                        recommended_skills = [
                            "Advanced Kotlin Programming: Master advanced Kotlin features like coroutines, higher-order functions, and DSLs for building complex Android apps.",
                            "Architectural Patterns: Understand architectural patterns like MVVM, MVP, or Clean Architecture for building scalable and maintainable Android apps.",
                            "Dependency Injection: Implement dependency injection using frameworks like Dagger or Koin to manage dependencies and improve app modularity.",
                            "Advanced UI/UX Techniques: Explore advanced UI/UX concepts such as animations, custom views, transitions, and gestures to enhance user experience.",
                            "Advanced Networking: Implement advanced networking features such as WebSocket communication, long-running background tasks, and real-time data synchronization.",
                            "Security Best Practices: Deepen your understanding of Android security principles and implement advanced security measures such as encryption, SSL pinning, and secure coding practices.",
                            "Performance Optimization: Optimize app performance by implementing techniques such as lazy loading, caching, memory management, and profiling.",
                            "Testing Automation: Develop skills in writing automated tests for Android apps using frameworks like Espresso, Mockito, and Robolectric for unit, integration, and UI testing.",
                            "Continuous Integration and Delivery (CI/CD): Set up CI/CD pipelines to automate app builds, testing, and deployment processes using tools like Jenkins, Travis CI, or CircleCI.",
                        ]
                    recommended_skills_selected = random.sample(recommended_skills, 3)
                    recommended_skills_not_in_resume = [skill for skill in recommended_skills_selected if skill not in resume_data['skills']]
                    rec_course = course_recommender(android_course)


                elif reco_field == 'IOS Development':



                    if cand_level == "Fresher" or cand_level == "NA" or cand_level == "Intermediate":
                        recommended_skills = [
                            "Swift Programming: Master the Swift programming language, including its syntax, data types, and advanced features.",
                            "User Interface (UI) Design: Learn to design user interfaces using Interface Builder and implement UI components programmatically.",
                            "iOS SDK: Familiarize yourself with the iOS Software Development Kit (SDK) and its components like view controllers, navigation controllers, and table views.",
                            "Database Management: Understand Core Data framework for data persistence and implement local data storage and retrieval in iOS apps.",
                            "Networking: Gain proficiency in making network requests and handling responses using libraries like URLSession or Alamofire.",
                            "User Authentication: Implement user authentication and authorization features using techniques like OAuth, Firebase Authentication, or JWT.",
                            "Version Control: Use Git for version control to track changes and collaborate with other developers effectively.",
                            "Testing and Debugging: Develop skills in testing iOS apps using XCTest framework for unit testing and debugging using Xcode tools.",
                            "Design Patterns: Learn common design patterns like MVC, MVVM, or VIPER for organizing code and improving maintainability.",
                        ]
                    elif cand_level =="Experienced" :
                        recommended_skills = [
                            "Advanced Swift Programming: Master advanced Swift features such as generics, optionals, closures, and concurrency for building complex iOS apps.",
                            "iOS Architecture Patterns: Understand advanced architectural patterns like VIPER, Redux, or Coordinator for building scalable and modular iOS apps.",
                            "Core Animation and Graphics: Explore Core Animation framework and advanced graphics techniques for creating fluid animations and custom UI components.",
                            "Advanced Networking: Implement advanced networking features such as WebSocket communication, long-polling, and real-time data synchronization.",
                            "Security Best Practices: Deepen your understanding of iOS security principles and implement advanced security measures such as encryption, SSL pinning, and secure coding practices.",
                            "Performance Optimization: Optimize app performance by implementing techniques such as lazy loading, caching, memory management, and asynchronous programming.",
                            "Advanced Debugging and Profiling: Develop skills in advanced debugging and profiling techniques using Xcode Instruments for identifying and resolving performance issues.",
                            "Continuous Integration and Delivery (CI/CD): Set up CI/CD pipelines to automate app builds, testing, and deployment processes using tools like Jenkins, Fastlane, or Bitrise.",
                            "Augmented Reality (AR) and Virtual Reality (VR): Explore ARKit and SceneKit frameworks for building immersive AR experiences, or Metal for high-performance VR apps.",
                        ]
                    recommended_skills_selected = random.sample(recommended_skills, 3)
                    recommended_skills_not_in_resume = [skill for skill in recommended_skills_selected if skill not in resume_data['skills']]
                    rec_course = course_recommender(ios_course)

                elif reco_field == 'UI-UX Development':


                    if cand_level == "Fresher" or cand_level == "NA" or cand_level == "Intermediate":

                            recommended_skills = [
                                "User Interface Design Principles: Master UI design principles such as hierarchy, balance, contrast, and alignment for creating visually appealing interfaces.",
                                "Wireframing and Prototyping: Learn tools like Sketch, Adobe XD, or Figma to create wireframes and interactive prototypes of digital products.",
                                "Typography and Color Theory: Understand typography principles and color theory to create effective and harmonious designs.",
                                "Responsive Design: Develop skills in designing responsive layouts that adapt to different screen sizes and devices.",
                                "Usability Testing: Conduct usability testing sessions to gather feedback and improve the user experience of digital products.",
                                "User Research: Gain knowledge of user research methodologies such as interviews, surveys, and user testing to inform design decisions.",
                                "Interaction Design: Design intuitive and user-friendly interactions using principles of affordance, feedback, and mapping.",
                                "Design Systems: Create and maintain design systems to ensure consistency and scalability across digital products.",
                                "UI Animation: Implement subtle animations and transitions to enhance user engagement and usability.",
                            ]
                    elif cand_level =="Experienced" :

                            recommended_skills = [
                                "Advanced Prototyping: Master advanced prototyping tools and techniques such as Principle, Proto.io, or InVision Studio for creating high-fidelity interactive prototypes.",
                                "Microinteractions: Design microinteractions to provide feedback and enhance the user experience in digital products.",
                                "User-Centered Design (UCD): Deepen your understanding of UCD principles and methodologies to prioritize user needs and preferences in the design process.",
                                "Accessibility: Ensure digital products are accessible to users with disabilities by following WCAG guidelines and implementing accessible design practices.",
                                "Design Thinking: Apply design thinking principles to solve complex problems and drive innovation in product development.",
                                "Data-Driven Design: Utilize data analytics and user feedback to inform design decisions and optimize the user experience.",
                                "Advanced User Research: Conduct advanced user research techniques such as ethnographic studies, diary studies, and A/B testing to gain deeper insights into user behavior.",
                                "UI/UX Automation: Implement UI/UX automation tools and processes to streamline design workflows and improve efficiency.",
                                "Cross-Platform Design: Develop skills in designing consistent and cohesive user experiences across multiple platforms and devices.",
                            ]
                    recommended_skills_selected = random.sample(recommended_skills, 3)
                    recommended_skills_not_in_resume = [skill for skill in recommended_skills_selected if skill not in resume_data['skills']]
                    rec_course = course_recommender(uiux_course)



                if recommended_skills_not_in_resume:
                    recommended_keywords = st_tags(label='# Some Advices for you.',
                                                text='Recommended advices generated from System',
                                                value=recommended_skills_not_in_resume,
                                                key='2')
                    st.markdown('''<h5 style='text-align: left; color: black;'>Adding these skills to your resume will boost the chances of getting a Job 🤖</h5>''', unsafe_allow_html=True)


            st.markdown("---")

            with st.expander("**How many pages should my resume be?**"):
                st.write("""


Candidates with less than 5 or 10 years of experience should stick to one page. If you're a senior candidate and have a lot of relevant experience, then a two-page resume is fine.

You should typically never go above 2 pages, and anything after the first two pages will generally not be read by hiring managers or recruiters.

If you decide to opt for a longer resume (e.g. 2 pages), please ensure that your most recent and critical achievements are on the first page of your resume, since most recruiters will stop reading at the first page.


""")
                
            with st.expander("**What's the ideal resume length?**"):
                st.write(""" 

The ideal resume length is a long debated topic. There's no right answer, but hiring managers and recruiters agree that your resume needs to be as concise as possible and should not contain unnecessary details. This generally means a one-page resume if you have less than 10 years of experience, and two pages if you have anything more than that. Cutting your resume down to one page forces you to leave in only the sections and bullets that matter.

In the USA, recruiters spend just 15-30 seconds per resume. While of course you won't get rejected if you send in 2 or more pages, recruiters will simply spend less time on each page and will miss your key achievements.

Note, of course, that expectations can vary by country. In the UK, a two page CV is more common while in Australia, for example, 2-3 page CVs aren't considered bad. Some countries that encourage longer resumes as standard may spend more time reviewing each resume. However, always keep in mind that the longer your resume, the less time will be spent by a recruiter reviewing each page. Adding in less relevant details will reduce the focus on your stronger accomplishments. If you're a student, you should always stick to a one page resume.

One page for 5-10 years of experience is a common rule of thumb. If you have more experience than that and that experience is relevant, then two pages is acceptable. It's important to stress that if you go over the one page guideline, please make sure that your experience is actually relevant to the job - don't include irrelevant coursework, hobbies or volunteering activities. And if you decide to opt for a longer resume (e.g. 2 pages), please ensure that your most recent and critical achievements are on the first page of your resume, since most recruiters will stop reading at the first page.

Our data suggests that top resumes have between 450 and 650 words for entry and mid-level hires, while up to 850 words for senior-level hires. This word count range often strikes the right balance between white space and depth.

""")

            st.markdown("---")



            st.subheader("**Resume Tips & Ideas **")
            resume_score = 0


            if 'resume_score' not in st.session_state:
                st.session_state.resume_score = 0

            def check_section(section_keywords):
                return any(keyword.lower() in resume_text.lower() for keyword in section_keywords)

            sections = {
                'Objective/Summary': ['objective', 'summary', 'career objective', 'professional summary'],
                'Education': ['education', 'school', 'college', 'university', 'bachelor', 'master', 'phd', 'degree', 'academic background'],
                'Experience': ['experience', 'career history', 'work history', 'employment history', 'professional experience', 'work experience'],
                'Internships': ['internship', 'intern', 'internships', 'intern experiences'],
                'Skills': ['skills', 'skill', 'technical skills', 'soft skills', 'proficiencies', 'competencies'],
                'Hobbies': ['hobbies', 'hobby', 'personal interests', 'leisure activities'],
                'Interests': ['interests', 'interest', 'personal interests', 'professional interests'],
                'Achievements': ['achievements', 'achievement', 'accomplishments', 'awards', 'honors', 'recognitions'],
                'Certifications': ['certifications', 'certification', 'certified', 'credentials', 'certificates'],
                'Projects': ['projects', 'project', 'portfolio', 'case studies', 'research projects']
            }

            section_scores = {
                'Experience': 25,
                'Education': 15,
                'Skills': 15,
                'Achievements': 10,
                'Projects': 10,
                'Certifications': 8,
                'Internships': 8,
                'Objective/Summary': 5,
                'Hobbies': 2,
                'Interests': 2
            }


            section_messages = {
                'Objective/Summary': (
                    "[+] Awesome! You have added Objective/Summary",
                    "[-] Please add your career objective. It will give your career intention to the recruiters."
                ),
                'Education': (
                    "[+] Awesome! You have added Education Details",
                    "[-] Please add Education. It will give your qualification level to the recruiter."
                ),
                'Experience': (
                    "[+] Awesome! You have added Experience",
                    "[-] Please add Experience. It will help you stand out from the crowd."
                ),
                'Internships': (
                    "[+] Awesome! You have added Internships",
                    "[-] Please add Internships. It will help you stand out from the crowd."
                ),
                'Skills': (
                    "[+] Awesome! You have added Skills",
                    "[-] Please add Skills. It will help you a lot."
                ),
                'Hobbies': (
                    "[+] Awesome! You have added your Hobbies",
                    "[-] Please add Hobbies. It will show your personality to the recruiters and assure them that you are a good fit for the role."
                ),
                'Interests': (
                    "[+] Awesome! You have added your Interests",
                    "[-] Please add Interests. It will show your interest beyond the job."
                ),
                'Achievements': (
                    "[+] Awesome! You have added your Achievements",
                    "[-] Please add Achievements. It will show that you are capable of the required position."
                ),
                'Certifications': (
                    "[+] Awesome! You have added your Certifications",
                    "[-] Please add Certifications. It will show that you have specialized for the required position."
                ),
                'Projects': (
                    "[+] Awesome! You have added your Projects",
                    "[-] Please add Projects. It will show that you have done work related to the required position."
                )
            }

            for section, keywords in sections.items():
                if check_section(keywords):
                    if f'{section}_checked' not in st.session_state:
                        st.session_state.resume_score += section_scores[section]
                        st.session_state[f'{section}_checked'] = True
                    st.markdown(f'''<h5 style='text-align: left; color: #1ed760;'>{section_messages[section][0]}</h5>''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''<h5 style='text-align: left; color: #000000;'>{section_messages[section][1]}</h5>''', unsafe_allow_html=True)

            st.subheader("**Resume Score 📝**")

            st.markdown(
                """
                <style>
                    .stProgress > div > div > div > div {
                        background-color: #d73b5c;
                    }
                </style>""",
                unsafe_allow_html=True,
            )

            my_bar = st.progress(0)

            for percent_complete in range(st.session_state.resume_score):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1)

            st.success(f'**Your Resume Writing Score: {st.session_state.resume_score}**')


            ts = time.time()
            cur_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            cur_time = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            timestamp = str(cur_date + '_' + cur_time)

            if resume_data['email'] is None or resume_data['email'] == "":
                email_value = act_mail
                st.error('Make sure that your resume contains a valid email address 🙂')
            else:
                email_value = resume_data['email']

            insert_data(str(sec_token), str(ip_add), host_name, dev_user, os_name_ver, latlong, city, state, country, act_name, act_mail, act_mob, resume_data['name'], email_value, str(resume_score), timestamp, str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']), str(recommended_skills), str(rec_course), pdf_name)


            st.subheader("**Bonus Video for Resume Writing Tips💡**")
            resume_vid = random.choice(resume_videos)
            st.video(resume_vid)



        else:
            st.error('Something went wrong..')                


    elif choice == 'Feedback': 

        ts = time.time()
        cur_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        cur_time = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        timestamp = str(cur_date + '_' + cur_time)


        with st.form("my_form", clear_on_submit=True):
            st.write("Feedback form")            
            feed_name = st.text_input('Name')
            feed_email = st.text_input('Email')
            feed_score = st.slider('Rate Us From 1 - 5', 1, 5)  
            comments = st.text_input('Comments')
            Timestamp = timestamp        
            submitted = st.form_submit_button("Submit")
            if submitted:
                if feed_email == "":
                    st.error("Email field cannot be empty.")
                elif not validate_email(feed_email):
                    st.error("Invalid email format.")
                elif feed_name == "":
                    st.error("Name field cannot be empty.")   
                else:
                    insertf_data(feed_name, feed_email, feed_score, comments, Timestamp)    
                    success_message = st.success("Thanks! Your Feedback was recorded.") 
                    st.balloons()   


        query = 'select * from user_feedback'        
        plotfeed_data = pd.read_sql(query, connection)                        


        labels = plotfeed_data.feed_score.unique()
        values = plotfeed_data.feed_score.value_counts()


    elif choice == 'Jobs':   

        max_category, _, _, _ = categorize_skills(resume_data)
        reco_field = max_category
        resume_text = pdf_reader(pdf_path)
        cand_level = determine_candidate_level(resume_text, resume_data)


        if cand_level == "Fresher" or cand_level == "NA":

                st.markdown(
                """
                ### Oops! 🚫

                It seems this page is exclusively reserved for the **Experienced** or **Intermediate** level. But hey, chin up! 😊
                Head on over to our user page where we've got a treasure trove of advice waiting just for you! 🎉

                Let's turn this little setback into a golden opportunity for growth! 🌱 So, grab your virtual gear, lace up those learning boots, and let's embark on an adventure to level up your skills! 💪

                Warmest regards,  
                Cyber cloud Team 🌟
                """
            )





        elif cand_level == "Experienced" or cand_level == "Intermediate" :


            if reco_field == 'Cyber Security': 

                job_message(reco_field)
                job_pills(cyber_jobs,reco_field)

                dataset = "./Datasets/cyber.csv"
                df = pd.read_csv(dataset)
                st.markdown("---")
                st.markdown("## Now let's play with statistics 🤹")

                tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(["Histogram" , "Scatter plot" , "Heatmap" , "Bar Chart" , "Pie Chart" , "Ring chart" , "Box plot"])
                numerical_columns = df.select_dtypes(include=np.number).columns.to_list()

                with tab1:
                    histogram_feature = st.selectbox(" Select feature to histogram" , numerical_columns , index=2)
                    fig_hist = px.histogram(df , x = histogram_feature)
                    st.plotly_chart(fig_hist)


                with tab2:

                    col1,col2,col3 = st.columns(3)

                    with col1:
                        x_columns = st.selectbox(" select column on x axis: " , numerical_columns  , index=2)
                    with col2:
                        y_columns = st.selectbox(" select column on y axis: " , numerical_columns , index=2)
                    with col3:
                        color = st.selectbox(" select column to be color " , df.columns , index=3)

                    fig_scatter = px.scatter(df , x = x_columns , y = y_columns,color =color )
                    st.plotly_chart(fig_scatter)


                with tab3:
                    data = pd.DataFrame(np.random.rand(10, 10), columns=[f'Col{i}' for i in range(10)])

                    st.markdown("## Heatmap")
                    corr_matrix = df.corr()
                    fig_heatmap = px.imshow(corr_matrix)
                    st.plotly_chart(fig_heatmap)

                with tab4:
                    z = df['job_title'].value_counts().head(10)
                    fig_bar = px.bar(z, x=z.index, y=z.values, color=z.index, text=z.values, labels={'x': 'Job Title', 'y': 'Count', 'text': 'Count'}, template='ggplot2', title='<b>Top 10 Roles in Cyber Security')
                    st.plotly_chart(fig_bar)

                    fig_bar_salary = px.bar(df.groupby('job_title', as_index=False)['salary_in_usd'].max().sort_values(by='salary_in_usd', ascending=False).head(10), x='job_title', y='salary_in_usd', color='job_title', labels={'job_title': 'Job Title', 'salary_in_usd': 'Salary in USD'}, text='salary_in_usd', template='seaborn', title='<b>Top 10 Highest Paid Job Titles')
                    st.plotly_chart(fig_bar_salary)

                with tab5:

                    fig_pie = px.pie(z, names=z.index, values=z.values, labels={'names': 'Job Title', 'values': 'Count'}, template='ggplot2')
                    st.plotly_chart(fig_pie)


                with tab6:

                    fig_pie_experience = px.pie(df.groupby('experience_level', as_index=False)['salary_in_usd'].count().sort_values(by='salary_in_usd', ascending=False).head(10), names='experience_level', values='salary_in_usd', color='experience_level', hole=0.7, labels={'experience_level': 'Experience Level', 'salary_in_usd': 'Count'}, template='ggplot2', title='<b>Experience Level in Cyber Security')
                    fig_pie_experience.update_layout(title_x=0.5, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
                    st.plotly_chart(fig_pie_experience)

                    fig_pie_employment = px.pie(df.groupby('employment_type', as_index=False)['salary_in_usd'].count().sort_values(by='salary_in_usd', ascending=False).head(10), names='employment_type', values='salary_in_usd', color='employment_type', hole=0.7, labels={'employment_type': 'Employment Type', 'salary_in_usd': 'Count'}, template='seaborn', title='<b>Employee Level in Cyber Security')
                    fig_pie_employment.update_layout(title_x=0.5)
                    st.plotly_chart(fig_pie_employment)

                with tab7:
                    fig_box = px.box(df, x='employment_type', y='salary_in_usd', color='employment_type', template='seaborn', labels={'employment_type': 'Employment Type', 'salary_in_usd': 'Salary in USD'}, title='<b>Cyber Security Salaries by type of employee')
                    st.plotly_chart(fig_box)

                    fig_box_company_size = px.box(df, x='company_size', y='salary_in_usd', color='company_size', template='ggplot2', labels={'company_size': 'Company Size', 'salary_in_usd': 'Salary in USD'}, title='<b>Cyber Security Salaries by Company Size')
                    st.plotly_chart(fig_box_company_size)

                st.markdown("---")    

                st.markdown(" # All Dataset 📈")
                n_rows = st.slider("Choose number of rows to display", min_value=5, max_value=len(df), step=1)
                columns_to_show = st.multiselect("Select columns to show", df.columns.to_list(), default=["job_title", "salary_in_usd"])

                display_option = st.selectbox("Choose display option", ["Regular", "Transpose"])

                if display_option == "Regular":
                    st.dataframe(df[:n_rows][columns_to_show])
                else:
                    st.dataframe(df[:n_rows][columns_to_show].T)

                st.markdown(" ## Bonus Video for Interview Tips💡")
                interview_vid = interview_videos[0]
                st.video(interview_vid)


            elif reco_field == 'Network Engineer':

                job_message(reco_field)
                job_pills(network_jobs,reco_field)
                job_statistics("Network Engineer")

                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = interview_videos[0]
                st.video(interview_vid)


            elif reco_field == 'Data Science':

                job_message(reco_field)
                job_pills(ds_jobs,reco_field)

                dataset = "./Datasets/ds_salaries.csv"
                df = pd.read_csv(dataset)
                st.markdown("## Now let's play with statistics 🤹")

                tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(["Histogram" , "Scatter plot" , "Heatmap" , "Bar Chart" , "Pie Chart" , "kernel density estimate" , "Box plot"])
                numerical_columns = df.select_dtypes(include=np.number).columns.to_list()

                with tab1:
                    histogram_feature = st.selectbox(" Select feature to histogram" , numerical_columns , index=2)
                    fig_hist = px.histogram(df , x = histogram_feature)
                    st.plotly_chart(fig_hist)


                with tab2:

                    col1,col2,col3 = st.columns(3)

                    with col1:
                        x_columns = st.selectbox(" select column on x axis: " , numerical_columns  , index=2)
                    with col2:
                        y_columns = st.selectbox(" select column on y axis: " , numerical_columns , index=2)
                    with col3:
                        color = st.selectbox(" select column to be color " , df.columns , index=3)

                    fig_scatter = px.scatter(df , x = x_columns , y = y_columns,color =color )
                    st.plotly_chart(fig_scatter)



                with tab3:

                    data = pd.DataFrame(np.random.rand(10, 10), columns=[f'Col{i}' for i in range(10)])

                    st.markdown("## Heatmap")
                    corr_matrix = df.corr()
                    fig_heatmap = px.imshow(corr_matrix)
                    st.plotly_chart(fig_heatmap)


                with tab4:

                    top15_job_titles = df['job_title'].value_counts()[:15]
                    fig = px.bar(y=top15_job_titles.values, x=top15_job_titles.index, text=top15_job_titles.values,
                                labels={'y': 'Count', 'x': 'Job Designations'}, title='<b> Top 15 Job Designations')
                    st.plotly_chart(fig)
                    salary_designation = df.nlargest(25, 'salary_in_usd')


                    fig = px.bar(x=salary_designation['job_title'], y=salary_designation['salary_in_usd'],
                                text=salary_designation['salary_in_usd'], color=salary_designation['salary_in_usd'])

                    fig.update_layout(xaxis_title="Job Designation", yaxis_title="Salaries",
                                    xaxis_tickangle=-45, title='<b> Top 25 Highest Salary by Designation 💸')
                    st.plotly_chart(fig)

                with tab5:
                    work_year = df['work_year'].value_counts()
                    fig = px.pie(values=work_year.values, names=work_year.index, title='<b> Work Year Distribution')
                    st.plotly_chart(fig)

                with tab6:

                    fig = px.histogram(df, x='salary_in_usd', nbins=20, histnorm='probability density', 
                                    marginal='rug', title='<b> Distribution Plot of Salary in USD')
                    st.plotly_chart(fig)

                with tab7:
                    fig = px.box(y=df['salary_in_usd'], title='<b> Salary Distribution in USD')
                    fig.update_layout(yaxis_title="Salary in USD")
                    st.plotly_chart(fig)


                st.markdown("---")    

                st.markdown(" # All Dataset 📈")
                n_rows = st.slider("Choose number of rows to display", min_value=5, max_value=len(df), step=1)
                columns_to_show = st.multiselect("Select columns to show", df.columns.to_list(), default=["job_title", "salary_in_usd"])

                display_option = st.selectbox("Choose display option", ["Regular", "Transpose"])

                if display_option == "Regular":
                    st.dataframe(df[:n_rows][columns_to_show])
                else:
                    st.dataframe(df[:n_rows][columns_to_show].T)

                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = interview_videos[0]
                st.video(interview_vid)

            elif reco_field == 'Web Development':

                job_message(reco_field)
                job_pills(web_jobs,reco_field)
                job_statistics('Web Developer')

                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = interview_videos[0]
                st.video(interview_vid)

            elif reco_field == 'Android Development':

                job_message(reco_field)
                job_pills(android_jobs,reco_field)
                job_statistics("Software Developer")

                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = interview_videos[0]
                st.video(interview_vid)

            elif reco_field == 'IOS Development':

                job_message(reco_field)
                job_pills(ios_jobs,reco_field)
                job_statistics("Software Developer")

                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = interview_videos[0]
                st.video(interview_vid)

            elif reco_field == 'UI-UX Development':

                job_message(reco_field)
                job_pills(uiux_jobs,reco_field)
                job_statistics("Software Developer")
                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = interview_videos[0]
                st.video(interview_vid)                




            else:

                st.error('please upload your resume first🙂')


    elif choice == 'Ai': 

        st.title("Ai magic 🚀")
        emoji = "🎈" 

        user_skills = resume_data.get('skills', [])
        user_skills = [skill.lower() for skill in user_skills]


        selected_skill = pills("Your current most critical Areas", user_skills, [emoji]*len(user_skills))

        if selected_skill:
            skill_description = skill_descriptions.get(selected_skill, "Description not available.")
            st.write(f"<span style='color:#F05126'>**{selected_skill}**</span>: {skill_description}", unsafe_allow_html=True)



        job_recommendations = get_ai_job_recommendations(user_skills)

        if job_recommendations:
            plot_ai_job_recommendations(job_recommendations)


        max_category, _, _, _ = categorize_skills(resume_data)
        reco_field = max_category

        generate_resume(resume_data)
        st.markdown("---")
        llm(reco_field)

    else:
            
        # Initialize session state variables for login
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'admin_user' not in st.session_state:
            st.session_state.admin_user = ''
        if 'admin_password' not in st.session_state:
            st.session_state.admin_password = ''

        # Admin login page
        if not st.session_state.logged_in:
            st.success('Welcome to Admin Side')

            ad_user = st.text_input("Username", key='admin_user')
            ad_password = st.text_input("Password", type='password', key='admin_password')

            if st.button('Login'):
                if ad_user == 'admin' and ad_password == 'admin':
                    st.session_state.logged_in = True
                    st.experimental_rerun() 
                    st.error("Invalid username or password")


        if st.session_state.logged_in:
            st.header("Admin Panel")
            cursor.execute('''SELECT ID, ip_add, resume_score, convert(Predicted_Field using utf8), 
                                convert(User_level using utf8), city, state, country from user_data''')
            datanalys = cursor.fetchall()
            plot_data = pd.DataFrame(datanalys, columns=['Idt', 'IP_add', 'resume_score', 'Predicted_Field', 
                                                        'User_Level', 'City', 'State', 'Country'])

            values = plot_data.Idt.count()
            st.success("Welcome 👨‍💻✨ ! Total %d " % values + " User's Have Used Our Tool : )")

            cursor.execute('''SELECT sec_token, ip_add, act_name, act_mail, act_mob, 
                                convert(Predicted_Field using utf8), Timestamp, Name, Email_ID, 
                                resume_score, Page_no, pdf_name, convert(User_level using utf8), 
                                convert(Actual_skills using utf8), convert(Recommended_skills using utf8), 
                                convert(Recommended_courses using utf8), city, state, country, latlong, 
                                os_name_ver, host_name, dev_user from user_data''')
            data = cursor.fetchall()

            st.header("**User's Data**")
            df = pd.DataFrame(data, columns=['Token', 'IP Address', 'Name', 'Mail', 'Mobile Number', 
                                            'Predicted Field', 'Timestamp', 'Predicted Name', 'Predicted Mail', 
                                            'Resume Score', 'Total Page',  'File Name', 'User Level', 'Actual Skills', 
                                            'Recommended Skills', 'Recommended Course', 'City', 'State', 'Country', 
                                            'Lat Long', 'Server OS', 'Server Name', 'Server User'])
            st.dataframe(df)

            if st.button('Download Report ✅'):
                st.markdown(get_csv_download_link(df, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)
            st.markdown("---")

            cursor.execute('''SELECT * from user_feedback''')
            data = cursor.fetchall()

            st.header("**User's Feedback Data**")
            df_feedback = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Feedback Score', 'Comments', 'Timestamp'])
            st.dataframe(df_feedback)

            st.subheader("**User Rating's**")
            fig = px.pie(df_feedback, values=df_feedback['Feedback Score'].value_counts(), names=df_feedback['Feedback Score'].unique(), 
                        title="Chart of User Rating Score From 1 - 5 🤗", color_discrete_sequence=px.colors.sequential.Aggrnyl)
            st.plotly_chart(fig)

            st.subheader("**Pie-Chart for Predicted Field Recommendation**")
            fig = px.pie(df, values=plot_data['Predicted_Field'].value_counts(), 
                        names=plot_data['Predicted_Field'].unique(), 
                        title='Predicted Field according to the Skills 👽', 
                        color_discrete_sequence=px.colors.sequential.Aggrnyl_r)
            st.plotly_chart(fig)

            st.subheader("**Pie-Chart for User's Experienced Level**")
            fig = px.pie(df, values=plot_data['User_Level'].value_counts(), 
                        names=plot_data['User_Level'].unique(), 
                        title="Pie-Chart 📈 for User's 👨‍💻 Experienced Level", 
                        color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig)

            st.subheader("**Pie-Chart for Resume Score**")
            fig = px.pie(df, values=plot_data['resume_score'].value_counts(), 
                        names=plot_data['resume_score'].unique(), 
                        title='From 1 to 100 💯', 
                        color_discrete_sequence=px.colors.sequential.Agsunset)
            st.plotly_chart(fig)

            st.subheader("**Pie-Chart for Users App Used Count**")
            fig = px.pie(df, values=plot_data['IP_add'].value_counts(), 
                        names=plot_data['IP_add'].unique(), 
                        title='Usage Based On IP Address 👥', 
                        color_discrete_sequence=px.colors.sequential.matter_r)
            st.plotly_chart(fig)

            st.subheader("**Pie-Chart for City**")
            fig = px.pie(df, values=plot_data['City'].value_counts(), 
                        names=plot_data['City'].unique(), 
                        title='Usage Based On City 🌆', 
                        color_discrete_sequence=px.colors.sequential.Jet)
            st.plotly_chart(fig)

            st.subheader("**Pie-Chart for Country**")
            fig = px.pie(df, values=plot_data['Country'].value_counts(), 
                        names=plot_data['Country'].unique(), 
                        title='Usage Based on Country 🌏', 
                        color_discrete_sequence=px.colors.sequential.Purpor_r)
            st.plotly_chart(fig)

            st.markdown("---")

            st.subheader("Manage Advertisements")
            ads = get_advertisements()
            df_ads = pd.DataFrame(ads)
            st.dataframe(df_ads)

            ad_id_to_delete = st.text_input("Enter the ID of the advertisement to delete")

            if st.button("Delete Advertisement"):
                if ad_id_to_delete:
                    try:
                        delete_advertisement(ad_id_to_delete)
                        st.success(f"Advertisement with ID {ad_id_to_delete} deleted successfully.")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Please enter an advertisement ID to delete.")


            st.sidebar.markdown("---")
            if st.sidebar.button("Logout"):
                st.session_state.logged_in = False
                st.experimental_rerun()


run()