"""
Sample course catalog for the Course Recommendation Assistant.

Each course is represented both as a plain dict (used by the custom tool
and for source-metadata lookups) and as a LangChain Document (used to
build the vector index for retrieval).
"""

from langchain_core.documents import Document

COURSES = [
    {
        "course_id": "SAP-ABAP-101",
        "course_name": "SAP ABAP Fundamentals",
        "skills": [
            "ABAP syntax",
            "SAP data dictionary",
            "Reports & modularization",
            "Debugging ABAP programs",
        ],
        "experience_level": "Beginner",
        "duration_hours": 30,
        "prerequisites": ["Basic programming knowledge"],
        "description": (
            "A foundational course covering core ABAP programming concepts, the SAP "
            "data dictionary, classic reports, and debugging techniques. Designed for "
            "developers who are new to the SAP ecosystem."
        ),
    },
    {
        "course_id": "AI-INTRO-101",
        "course_name": "Introduction to Artificial Intelligence",
        "skills": [
            "AI concepts",
            "Machine learning basics",
            "Neural network fundamentals",
            "AI use-case identification",
        ],
        "experience_level": "Beginner",
        "duration_hours": 15,
        "prerequisites": ["None"],
        "description": (
            "A gentle, non-mathematical introduction to what AI and machine learning "
            "are, how they differ, common algorithms, and how businesses apply AI "
            "today. Ideal for developers or analysts with zero prior AI exposure."
        ),
    },
    {
        "course_id": "SAP-BAI-201",
        "course_name": "SAP Business AI Foundations",
        "skills": [
            "SAP Business AI portfolio",
            "Joule copilot",
            "Embedded AI in SAP S/4HANA",
            "AI ethics in SAP",
        ],
        "experience_level": "Intermediate",
        "duration_hours": 20,
        "prerequisites": ["Introduction to Artificial Intelligence", "Basic SAP navigation knowledge"],
        "description": (
            "Covers SAP's Business AI portfolio, including Joule, embedded AI "
            "scenarios in S/4HANA, and how generative and predictive AI are packaged "
            "for SAP customers. Bridges general AI knowledge with SAP-specific "
            "application."
        ),
    },
    {
        "course_id": "ML-PY-201",
        "course_name": "Machine Learning with Python",
        "skills": [
            "Python for ML",
            "scikit-learn",
            "Model training & evaluation",
            "Feature engineering",
        ],
        "experience_level": "Intermediate",
        "duration_hours": 40,
        "prerequisites": ["Introduction to Artificial Intelligence", "Basic Python"],
        "description": (
            "Hands-on course building classification and regression models with "
            "Python and scikit-learn. Covers the full ML workflow from data "
            "preparation to evaluation."
        ),
    },
    {
        "course_id": "GENAI-ENT-301",
        "course_name": "Generative AI for Enterprise Applications",
        "skills": [
            "LLM fundamentals",
            "Prompt engineering",
            "RAG architecture",
            "Enterprise AI governance",
        ],
        "experience_level": "Intermediate",
        "duration_hours": 25,
        "prerequisites": ["Introduction to Artificial Intelligence"],
        "description": (
            "Explores how large language models are applied inside enterprises: "
            "prompt engineering, retrieval-augmented generation, and governance "
            "considerations for deploying generative AI safely at scale."
        ),
    },
    {
        "course_id": "SAP-BTP-AI-401",
        "course_name": "SAP BTP and AI Integration",
        "skills": [
            "SAP BTP AI services",
            "Building custom AI extensions",
            "SAP AI Core",
            "Integrating LLMs with SAP data",
        ],
        "experience_level": "Advanced",
        "duration_hours": 35,
        "prerequisites": ["SAP Business AI Foundations", "Generative AI for Enterprise Applications"],
        "description": (
            "An advanced, hands-on course on building and deploying custom AI "
            "extensions on SAP Business Technology Platform, using SAP AI Core and "
            "integrating LLMs with live SAP data. Capstone course for the SAP "
            "Business AI track."
        ),
    },
]


def build_documents():
    """Convert the COURSES catalog into LangChain Documents for embedding."""
    docs = []
    for c in COURSES:
        page_content = (
            f"Course name: {c['course_name']}\n"
            f"Skills taught: {', '.join(c['skills'])}\n"
            f"Experience level: {c['experience_level']}\n"
            f"Duration: {c['duration_hours']} hours\n"
            f"Prerequisites: {', '.join(c['prerequisites'])}\n"
            f"Description: {c['description']}"
        )
        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "course_id": c["course_id"],
                    "course_name": c["course_name"],
                    "experience_level": c["experience_level"],
                    "duration_hours": c["duration_hours"],
                    "prerequisites": ", ".join(c["prerequisites"]),
                    "source": f"course_catalog::{c['course_id']}",
                },
            )
        )
    return docs
