"""
Streamlit front end for the Course Recommendation Assistant.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st

from rag_engine import CourseRecommendationAssistant
from courses_data import COURSES

st.set_page_config(page_title="Course Recommendation Assistant", page_icon="🎓", layout="wide")

st.title("🎓 Course Recommendation Assistant")
st.caption("RAG-powered course advisor built with LangChain + Google Gemini")

# ---------------------------------------------------------------- Sidebar --
with st.sidebar:
    st.header("Setup")
    api_key_input = st.text_input(
        "Google API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Get a free key from https://aistudio.google.com/app/apikey",
    )

    if st.button("Reset conversation"):
        if "assistant" in st.session_state:
            st.session_state.assistant.reset_history()
        st.session_state.chat = []
        st.rerun()

    st.divider()
    st.subheader("📚 Course Catalog")
    for c in COURSES:
        with st.expander(c["course_name"]):
            st.write(f"**Level:** {c['experience_level']}")
            st.write(f"**Duration:** {c['duration_hours']} hours")
            st.write(f"**Skills:** {', '.join(c['skills'])}")
            st.write(f"**Prerequisites:** {', '.join(c['prerequisites'])}")

# ------------------------------------------------------------- Init state --
if "chat" not in st.session_state:
    st.session_state.chat = []

if not api_key_input:
    st.info("Enter your Google API key in the sidebar to get started.")
    st.stop()

if "assistant" not in st.session_state or st.session_state.get("_key") != api_key_input:
    with st.spinner("Indexing course catalog..."):
        try:
            st.session_state.assistant = CourseRecommendationAssistant(
                google_api_key=api_key_input
            )
            st.session_state._key = api_key_input
        except Exception as e:
            st.error(f"Failed to initialize assistant: {e}")
            st.stop()

# ------------------------------------------------------------- Chat replay --
for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.write(turn["content"])
        else:
            st.write(turn["content"])

# ----------------------------------------------------------------- Input --
question = st.chat_input(
    "e.g. I'm an SAP ABAP developer with no AI experience. Which course should I take first?"
)

if question:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.assistant.ask(question)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        confidence_pct = f"{result.confidence:.0%}"
        st.markdown(f"**Confidence:** {confidence_pct}")
        st.write(result.reason)

        if result.recommended_courses:
            st.subheader("Recommended Courses")
            for rc in result.recommended_courses:
                with st.container(border=True):
                    st.markdown(f"### {rc.course_name}")
                    st.write(f"**Level:** {rc.experience_level}  |  **Duration:** {rc.duration}")
                    st.write(f"**Skills covered:** {', '.join(rc.skills_covered)}")
                    st.write(f"**Why:** {rc.match_reason}")
                    st.caption(
                        f"Source: {rc.source.source}  (course_id={rc.source.course_id})"
                    )

        if result.prerequisites:
            st.subheader("Prerequisites")
            for p in result.prerequisites:
                st.write(f"- {p}")

        if result.learning_sequence:
            st.subheader("Learning Sequence")
            st.write(" → ".join(result.learning_sequence))

        col1, col2 = st.columns(2)
        with col1:
            if result.total_learning_hours:
                st.metric("Total Learning Hours", f"{result.total_learning_hours:.0f}h")
        with col2:
            st.metric("Confidence", confidence_pct)

        if result.tool_output:
            st.caption(f"🔧 calculate_total_learning_hours tool output: {result.tool_output}")

    st.session_state.chat.append({"role": "assistant", "content": result.reason})
