import pdfplumber
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =====================================================
# SKILL DATABASE
# =====================================================

SKILLS = [

    # Programming
    "python",
    "java",
    "c",
    "c++",
    "javascript",

    # Databases
    "sql",
    "mysql",
    "postgresql",
    "mongodb",

    # ML / AI
    "machine learning",
    "deep learning",
    "data science",
    "data analysis",
    "nlp",
    "computer vision",

    # Libraries
    "tensorflow",
    "keras",
    "pytorch",
    "scikit-learn",
    "opencv",

    # Cloud
    "aws",
    "azure",
    "gcp",

    # Frameworks
    "django",
    "flask",
    "fastapi",
    "streamlit",

    # Tools
    "git",
    "github",
    "docker",
    "linux",

    # BI
    "power bi",
    "tableau",
    "excel",

    # Gen AI
    "llm",
    "langchain",
    "rag",
    "generative ai"
]


# =====================================================
# EDUCATION KEYWORDS
# =====================================================

EDUCATION_KEYWORDS = [

    "b.tech",
    "btech",
    "b.e",
    "be",
    "m.tech",
    "mtech",
    "bca",
    "mca",
    "bsc",
    "msc",
    "mba",
    "phd"
]


# =====================================================
# PROJECT KEYWORDS
# =====================================================

PROJECT_KEYWORDS = [

    "project",
    "projects",
    "developed",
    "built",
    "implemented",
    "designed",
    "created"
]


# =====================================================
# CERTIFICATION KEYWORDS
# =====================================================

CERTIFICATION_KEYWORDS = [

    "certificate",
    "certification",
    "nptel",
    "coursera",
    "udemy",
    "edunet",
    "aws certified",
    "google cloud",
    "oracle certified",
    "microsoft certified",
    "tensorflow developer"
]


# =====================================================
# PDF EXTRACTION
# =====================================================

def extract_text_from_pdf(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    return text


# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        ' ',
        text
    )

    tokens = text.split()

    stop_words = set(
        stopwords.words("english")
    )

    tokens = [

        word

        for word in tokens

        if word not in stop_words

    ]

    lemmatizer = WordNetLemmatizer()

    tokens = [

        lemmatizer.lemmatize(word)

        for word in tokens

    ]

    return " ".join(tokens)


# =====================================================
# SKILL EXTRACTION
# =====================================================

def extract_skills(text):

    text = text.lower()

    detected_skills = []

    for skill in SKILLS:

        pattern = r'\b' + re.escape(skill) + r'\b'

        if re.search(pattern, text):

            detected_skills.append(skill)

    return sorted(
        list(set(detected_skills))
    )


# =====================================================
# EDUCATION EXTRACTION
# =====================================================

def extract_education(text):

    text = text.lower()

    education = []

    for degree in EDUCATION_KEYWORDS:

        pattern = r'\b' + re.escape(degree) + r'\b'

        if re.search(pattern, text):

            education.append(degree)

    return list(set(education))


# =====================================================
# EXPERIENCE EXTRACTION
# =====================================================

def extract_experience(text):

    text = text.lower()

    pattern = r'(\d+)\s*\+?\s*(year|years)'

    matches = re.findall(
        pattern,
        text
    )

    if matches:

        years = [

            int(match[0])

            for match in matches

        ]

        return max(years)

    return 0


# =====================================================
# PROJECT COUNT
# =====================================================

def count_projects(text):

    text = text.lower()

    count = 0

    for keyword in PROJECT_KEYWORDS:

        count += text.count(keyword)

    return count


# =====================================================
# CERTIFICATION EXTRACTION
# =====================================================

def extract_certifications(text):

    text = text.lower()

    certifications = []

    for cert in CERTIFICATION_KEYWORDS:

        if cert in text:

            certifications.append(cert)

    return list(set(certifications))


# =====================================================
# ATS SCORE
# =====================================================

def calculate_ats_score(

    similarity_score,
    skill_score,
    education_score,
    experience_score,
    project_score,
    certification_score

):

    ats_score = (

        (0.40 * similarity_score)

        +

        (0.25 * skill_score)

        +

        (0.10 * education_score)

        +

        (0.10 * experience_score)

        +

        (0.10 * project_score)

        +

        (0.05 * certification_score)

    )

    return round(
        ats_score,
        2
    )


# =====================================================
# CANDIDATE SUMMARY
# =====================================================

def generate_candidate_summary(

    candidate_name,
    ats_score,
    matched_skills,
    missing_skills

):

    summary = {}

    summary["Candidate"] = candidate_name

    summary["Matched Skills"] = matched_skills

    summary["Missing Skills"] = missing_skills

    if ats_score >= 85:

        summary["Recommendation"] = (
            "Strong Candidate"
        )

        summary["Interview Status"] = (
            "Proceed to Technical Round"
        )

    elif ats_score >= 70:

        summary["Recommendation"] = (
            "Potential Candidate"
        )

        summary["Interview Status"] = (
            "Needs Further Review"
        )

    else:

        summary["Recommendation"] = (
            "Not Recommended"
        )

        summary["Interview Status"] = (
            "Reject"
        )

    return summary