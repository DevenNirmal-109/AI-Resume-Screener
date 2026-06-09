import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    extract_text_from_pdf,
    clean_text,
    extract_skills,
    extract_education,
    extract_experience,
    extract_certifications,
    count_projects,
    calculate_ats_score,
    generate_candidate_summary
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = SentenceTransformer(
    "./models/all-MiniLM-L6-v2"
)

# =====================================================
# TITLE
# =====================================================

st.title("📄 AI Resume Screening & Ranking System")

st.markdown(
    """
    Upload multiple resumes and a Job Description.
    
    The system will:
    
    ✅ Rank candidates
    
    ✅ Extract skills
    
    ✅ Identify missing skills
    
    ✅ Generate AI recommendations
    """
)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_resumes = st.file_uploader(
    "Upload Candidate Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

uploaded_jd = st.file_uploader(
    "Upload Job Description",
    type=["pdf"]
)

# =====================================================
# MAIN PROCESSING
# =====================================================

if uploaded_resumes and uploaded_jd:

    st.success("Files Uploaded Successfully")

    # =================================================
    # JOB DESCRIPTION PROCESSING
    # =================================================

    jd_text = extract_text_from_pdf(
        uploaded_jd
    )

    cleaned_jd = clean_text(
        jd_text
    )

    jd_embedding = model.encode(
        cleaned_jd
    )

    jd_skills = extract_skills(
        cleaned_jd
    )

    st.subheader("📋 Required Skills")

    if jd_skills:

        st.info(
            ", ".join(jd_skills)
        )

    else:

        st.warning(
            "No skills detected in JD."
        )

    # =================================================
    # STORE RESULTS
    # =================================================

    results = []

    st.subheader(
        "📄 Resume Analysis"
    )

    # =================================================
    # PROCESS EACH RESUME
    # =================================================

    for resume in uploaded_resumes:

        # =====================================
        # Resume Processing
        # =====================================

        resume_text = extract_text_from_pdf(
            resume
        )

        cleaned_resume = clean_text(
            resume_text
        )

        resume_embedding = model.encode(
            cleaned_resume
        )

        # =====================================
        # Extract Information
        # =====================================

        resume_skills = extract_skills(
            resume_text
        )

        resume_education = extract_education(
            resume_text
        )

        resume_experience = extract_experience(
            resume_text
        )

        resume_certifications = (
            extract_certifications(
                resume_text
            )
        )

        projects_count = count_projects(
            resume_text
        )

        # =============================================
        # SIMILARITY SCORE
        # =============================================

        similarity_score = cosine_similarity(

            [jd_embedding],
            [resume_embedding]

        )[0][0]

        similarity_score = round(
            similarity_score * 100,
            2
        )

        # =============================================
        # SKILL MATCHING
        # =============================================

        matched_skills = list(
            set(jd_skills).intersection(
                set(resume_skills)
            )

        )

        missing_skills = list(

            set(jd_skills) -
            set(resume_skills)

        )

        if len(jd_skills) > 0:

            skill_score = round(

                (
                    len(matched_skills)
                    /
                    len(jd_skills)
                ) * 100,

                2

            )

        else:

            skill_score = 0
        # ============================================
        # Education score
        # ============================================
        jd_education = extract_education(
            jd_text
        )

        education_score = 100

        if jd_education:

            if len(

                set(jd_education).intersection(
                    set(resume_education)
                )

            ) == 0:

                education_score = 0
                
        # ============================================
        # Experience score
        # ============================================
        jd_experience = extract_experience(
            jd_text
        )

        if jd_experience == 0:

            experience_score = 100

        elif resume_experience >= jd_experience:

            experience_score = 100

        else:

            experience_score = round(

                (
                    resume_experience
                    /
                    jd_experience
                ) * 100,

                2

            )
        
        # ============================================
        # Project score
        # ============================================
        if projects_count >= 5:

            project_score = 100

        elif projects_count >= 3:

            project_score = 80

        elif projects_count >= 1:

            project_score = 60

        else:

            project_score = 0
        
        # ============================================
        # Certification score
        # ============================================
        certification_count = len(
            resume_certifications
        )

        if certification_count >= 5:

            certification_score = 100

        elif certification_count >= 3:

            certification_score = 80

        elif certification_count >= 1:

            certification_score = 60

        else:

            certification_score = 0
            
        # ============================================
        # ATS Score
        # ============================================
        ats_score = calculate_ats_score(

            similarity_score,

            skill_score,

            education_score,

            experience_score,

            project_score,

            certification_score

        )
        # ============================================
        # Candidate Summary
        # ============================================
        candidate_summary = generate_candidate_summary(

            resume.name,

            ats_score,

            matched_skills,

            missing_skills

        )
        if ats_score >= 85:

            category = "Excellent Fit"

        elif ats_score >= 70:

            category = "Good Fit"

        elif ats_score >= 55:

            category = "Average Fit"

        else:

            category = "Poor Fit"

        # =============================================
        # SAVE RESULT
        # =============================================

        results.append({

            "Candidate": resume.name,

            "ATS Score (%)": ats_score,

            "Similarity (%)": similarity_score,

            "Skill Match (%)": skill_score,

            "Projects": projects_count,

            "Experience (Years)": resume_experience,

            "Certificates": certification_count,

            "Category": category

        })

        # =============================================
        # DISPLAY DETAILS
        # =============================================

        with st.expander(
            f"📄 {resume.name}"
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Similarity Score",
                    f"{similarity_score}%"
                )

            with col2:

                st.metric(
                    "Skill Match",
                    f"{skill_score}%"
                )

            st.info(
                f"Category: {category}"
            )

            # -----------------------------------------

            st.write(
                "## Matched Skills"
            )

            if matched_skills:

                for skill in matched_skills:

                    st.success(skill)

            else:

                st.warning(
                    "No matching skills found."
                )

            # -----------------------------------------

            st.write(
                "## Missing Skills"
            )

            if missing_skills:

                for skill in missing_skills:

                    st.error(skill)

            else:

                st.success(
                    "No missing skills."
                )

            # -----------------------------------------

            st.write(
                "## AI Candidate Summary"
            )

            st.write(

                f"**Recommendation:** "
                f"{candidate_summary['Recommendation']}"

            )

            st.write(

                f"**Interview Status:** "
                f"{candidate_summary['Interview Status']}"

            )

            # -----------------------------------------

            st.write(
                "## Resume Skills"
            )

            st.write(
                ", ".join(resume_skills)
            )

    # =================================================
    # SORT RESULTS
    # =================================================

    results = sorted(

        results,

        key=lambda x:
        x["Similarity (%)"],

        reverse=True

    )

    # =================================================
    # RANKING TABLE
    # =================================================

    st.subheader(
        "🏆 Candidate Rankings"
    )

    ranking_df = pd.DataFrame(
        results
    )

    st.dataframe(
        ranking_df,
        use_container_width=True
    )

    # =================================================
    # TOP CANDIDATE
    # =================================================

    top_candidate = results[0]

    st.success(

        f"🏆 Best Candidate: "
        f"{top_candidate['Candidate']} "
        f"({top_candidate['Similarity (%)']}%)"

    )