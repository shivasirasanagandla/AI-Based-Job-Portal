from tkinter import Image
from torch.serialization import add_safe_globals
from transformers.models.bert import BertConfig
import PyPDF2
import streamlit as st
import pytesseract
from difflib import get_close_matches
import pdfplumber
import spacy
from docx import Document
import re
import torch
import shap
from shap.maskers import Text
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt

# Custom Adversarial Model Definition
class MainModelWithAdversary(torch.nn.Module):
    def __init__(self, bert_model, num_labels, lambda_):
        super(MainModelWithAdversary, self).__init__()
        self.bert = bert_model
        self.classifier = torch.nn.Linear(bert_model.config.hidden_size, num_labels)
        self.adversary = torch.nn.Linear(bert_model.config.hidden_size, 1)
        self.lambda_ = lambda_

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        sequence_output = hidden_states[-1]
        pooled_output = sequence_output[:, 0, :]  # [CLS] token output

        label_preds = self.classifier(pooled_output)
        sensitive_preds = self.adversary(pooled_output)

        return label_preds, sensitive_preds


# SHAP Model Wrapper for Explainability
class SHAPModelWrapper:
    def __init__(self, model, tokenizer, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for k, v in inputs.items()}
        with torch.no_grad():
            logits, _ = self.model(**inputs)
        return logits.cpu().numpy()


# Loading Adversarial Model
@st.cache_resource
def load_adversarial_model():
    add_safe_globals([BertConfig])
    checkpoint = torch.load("adversarial_debiasing_model.pth", map_location=torch.device("cpu"))
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=checkpoint["num_labels"])
    model = MainModelWithAdversary(bert_model, num_labels=checkpoint["num_labels"], lambda_=checkpoint["lambda_"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


model, tokenizer = load_adversarial_model()

# Predefine the role-to-class mapping
role_to_class = {
    "python developer": 1, "data scientist": 2, "data analyst": 3,
    "web designing": 4, "business analyst": 5, "business development": 6,
    "hr": 7, "public relations": 8, "network security engineer": 9,
    "software engineer": 10, "devops engineer": 11, "automation tester": 12,
    "ui/ux designer": 13
}

# Keyword-Based Post-Processing
keywords = {
    "python developer": ["python", "django", "flask", "machine learning", "pandas", "numpy"],
    "data scientist": ["data science", "machine learning", "deep learning", "statistics", "ai", "nlp"],
    "data analyst": ["data analysis", "excel", "power bi", "tableau", "sql", "analytics"],
    "web designing": ["html", "css", "javascript", "ui/ux", "adobe xd", "figma", "responsive design"],
    "business analyst": ["business analysis", "process improvement", "requirements gathering", "agile", "scrum"],
    "business development": ["lead generation", "sales", "marketing", "client relationships", "negotiation"],
    "hr": ["human resources", "recruitment", "talent acquisition", "employee engagement", "hr policies"],
    "public relations": ["media", "communication", "press releases", "public relations", "crisis management"],
    "network security engineer": ["networking", "security", "firewalls", "routing", "switching", "vpn", "cisco"],
    "software engineer": ["software development", "java", "c++", "system design", "git", "agile"],
    "devops engineer": ["devops", "ci/cd", "jenkins", "kubernetes", "docker", "aws", "linux"],
    "automation tester": ["automation testing", "selenium", "test cases", "regression testing", "junit", "cypress"],
    "ui/ux designer": ["ui design", "ux design", "prototyping", "adobe xd", "figma", "user experience", "wireframing"]
}
# Resume Text Extraction Functions
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    sections = {
        "Name": "", "Contact": "", "GitHub": "", "LinkedIn": "", "Email": "", 
        "Education": [], "Internship Experience": [], "Skills": [], "Projects": [], 
        "Certification": [], "Profile": [], "Objective":[], "Summary":[], "Work Experience":[], "Tools&Technology": []
    }
    current_section = None

    phone_pattern = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}")
    name_pattern = re.compile(r"^(?!.*@)([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)")
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    github_pattern = re.compile(r"https?://github\.com/[\w-]+")
    linkedin_pattern = re.compile(r"https?://(www\.)?linkedin\.com/in/[\w-]+")

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            lines = page_text.split("\n")
            for line in lines:
                line = line.strip()
                
                if not sections["Name"] and name_pattern.match(line) and "Intern" not in line:
                    sections["Name"] = name_pattern.match(line).group()
                elif not sections["Contact"] and phone_pattern.search(line):
                    sections["Contact"] = phone_pattern.search(line).group()
                elif not sections["GitHub"] and github_pattern.search(line):
                    sections["GitHub"] = github_pattern.search(line).group()
                elif not sections["LinkedIn"] and linkedin_pattern.search(line):
                    sections["LinkedIn"] = linkedin_pattern.search(line).group()
                elif not sections["Email"] and email_pattern.search(line):
                    sections["Email"] = email_pattern.search(line).group()
                else:
                    for sec in sections.keys():
                        if re.search(rf"(?i)^{sec}[:\s]*$", line):
                            current_section = sec
                            if not isinstance(sections[current_section], list):
                                sections[current_section] = []
                            break
                    else:
                        if current_section:
                            sections[current_section].append(line)
    
    formatted_text = "\n\n".join([
        f"**{section}:**\n" + ("\n".join(content) if isinstance(content, list) else content) 
        for section, content in sections.items() if content
    ])
    
    return formatted_text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    sections = {
        "Name": "", "Contact": "", "GitHub": "", "LinkedIn": "", "Email": "", 
        "Education": [], "Internship Experience": [], "Skills": [], "Projects": [], 
        "Certification": [], "Profile": [], "Objective":[], "Summary":[], "Work Experience":[], "Tools&Technology": []
    }
    current_section = None
    
    phone_pattern = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}")
    name_pattern = re.compile(r"^(?!.*@)([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)")
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    github_pattern = re.compile(r"https?://github\.com/[\w-]+")
    linkedin_pattern = re.compile(r"https?://(www\.)?linkedin\.com/in/[\w-]+")
    
    for para in doc.paragraphs:
        text = para.text.strip()
        
        if not sections["Name"] and name_pattern.match(text) and "Intern" not in text:
            sections["Name"] = name_pattern.match(text).group()
        elif not sections["Contact"] and phone_pattern.search(text):
            sections["Contact"] = phone_pattern.search(text).group()
        elif not sections["GitHub"] and github_pattern.search(text):
            sections["GitHub"] = github_pattern.search(text).group()
        elif not sections["LinkedIn"] and linkedin_pattern.search(text):
            sections["LinkedIn"] = linkedin_pattern.search(text).group()
        elif not sections["Email"] and email_pattern.search(text):
            sections["Email"] = email_pattern.search(text).group()
        else:
            for sec in sections.keys():
                if re.search(rf"(?i)^{sec}[:\s]*$", text):
                    current_section = sec
                    if not isinstance(sections[current_section], list):
                        sections[current_section] = []
                    break
            else:
                if current_section:
                    sections[current_section].append(text)
    
    formatted_text = "\n\n".join([
        f"**{section}:**\n" + ("\n".join(content) if isinstance(content, list) else content) 
        for section, content in sections.items() if content
    ])
    
    return formatted_text


# Preprocessing and Tokenization
def preprocess_and_tokenize(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return inputs


# Predicting Suitability
def predict_suitability(model, inputs):
    with torch.no_grad():
        input_ids = inputs["input_ids"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        attention_mask = inputs["attention_mask"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class, logits


def get_missing_skills(role, resume_text):
    """Identify missing skills for a specific role based on the resume."""
    resume_text_lower = resume_text.lower()
    required_skills = keywords.get(role, [])
    missing_skills = [skill for skill in required_skills if skill not in resume_text_lower]
    return missing_skills


# Streamlit UI
st.title("AI-Powered Resume Screening and Feedback System")

st.subheader("Job Description (JD)")
job_description = st.text_area("Enter the JD for the role you're applying for:")

uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file and job_description:
    # Extract Resume Text
    if uploaded_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format!")
        st.stop()

    # Ensure resume_text is a string
    if not isinstance(resume_text, str) or not resume_text.strip():
        st.error("Invalid or empty resume text. Please upload a valid resume.")
        st.stop()

    st.subheader("Extracted Resume Text")
    st.write(resume_text)

    # Selecting Role
    role = st.selectbox("Select the role you are applying for:", list(role_to_class.keys()))

    if role:
        normalized_role = role.lower()  # Converting the input to lowercase for case-insensitive matching
        selected_class = role_to_class.get(normalized_role)
        if selected_class:
            st.write(f"Selected Role: {role}, Mapped Class: {selected_class}")

            # Predict Suitability
            inputs = preprocess_and_tokenize(resume_text)
            predicted_class, logits = predict_suitability(model, inputs)

            # Refine decision
            refined_class = role if role in keywords and any(
                keyword in resume_text.lower() for keyword in keywords[role]
            ) else "Other"

            decision = "Selected" if refined_class == role.lower() else "Rejected"
            st.subheader("Prediction Result")
            st.write(f"Decision: {decision}")

            # Feedback Section
            st.subheader("Feedback")
            missing_skills = get_missing_skills(normalized_role, resume_text)
            if decision == "Rejected" and missing_skills:
                feedback = (
                    f"Your resume does not meet the job criteria. "
                    f"Consider improving your skills in the following areas: {', '.join(missing_skills)}."
                )
            elif decision == "Rejected":
                feedback = "Your resume does not meet the job criteria, but no specific missing skills were identified."
            else:
                feedback = "Your resume aligns well with the requirements."
            st.write(feedback)
else:
    if not job_description:
        st.warning("Please enter a Job Description (JD).")
