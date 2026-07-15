"""
RAG engine for the Course Recommendation Assistant.

Pipeline:
  1. Embed the course catalog with Google Gemini embeddings into a Chroma
     vector store.
  2. Retrieve the most relevant courses for the learner's question.
  3. Ask Gemini (via ChatGoogleGenerativeAI.with_structured_output) to
     produce a RecommendationResponse grounded only in the retrieved
     context.
  4. Attach accurate source metadata from the catalog.
  5. Call the custom calculate_total_learning_hours tool to compute the
     total duration of the recommended learning sequence.
  6. Track conversation history across turns.
"""

import os
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from courses_data import build_documents, COURSES
from models import RecommendationResponse, SourceMetadata
from tools import calculate_total_learning_hours, calculate_total_hours_raw

SYSTEM_PROMPT = """You are an expert learning advisor for a corporate training \
catalog focused on SAP and Artificial Intelligence courses.

Use ONLY the course context provided below to make recommendations. Never \
invent a course that is not present in the context. If nothing in the \
context is relevant, say so honestly in `reason` and return an empty \
recommended_courses list.

Course context:
{context}

Given the learner's background, existing skills, and stated goal, recommend \
the most relevant course or ordered sequence of courses. Take into account \
their current experience level to decide whether foundational courses are \
needed before more advanced ones, and use each course's stated \
prerequisites to build a sensible learning_sequence.

Respond with a structured recommendation containing:
- recommended_courses: the specific course(s) to take, each with a clear match_reason
- reason: your overall reasoning for the recommendation
- prerequisites: prerequisites the learner still needs, if any
- learning_sequence: ordered list of course names, from current level to goal
- confidence: your confidence in this recommendation, between 0 and 1
"""


class CourseRecommendationAssistant:
    def __init__(self, google_api_key: Optional[str] = None):
        api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "A Google API key is required. Set the GOOGLE_API_KEY "
                "environment variable or pass google_api_key explicitly."
            )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2
        )
        self.structured_llm = self.llm.with_structured_output(RecommendationResponse)

        self.documents = build_documents()
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            collection_name="course_catalog",
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )

        # Bonus requirement: conversation history, kept in memory for the
        # lifetime of this assistant instance / Streamlit session.
        self.history: List = []

        self._course_by_name = {c["course_name"].lower(): c for c in COURSES}

    def _format_context(self, docs) -> str:
        return "\n\n---\n\n".join(d.page_content for d in docs)

    def _attach_sources(self, response: RecommendationResponse) -> RecommendationResponse:
        """Overwrite whatever metadata the model guessed with the ground
        truth from the catalog, so source metadata is always accurate."""
        for rc in response.recommended_courses:
            match = self._course_by_name.get(rc.course_name.strip().lower())
            if match:
                rc.source = SourceMetadata(
                    course_id=match["course_id"],
                    course_name=match["course_name"],
                    source=f"course_catalog::{match['course_id']}",
                    experience_level=match["experience_level"],
                    duration=f"{match['duration_hours']} hours",
                )
                rc.duration = f"{match['duration_hours']} hours"
                rc.experience_level = match["experience_level"]
        return response

    def ask(self, question: str) -> RecommendationResponse:
        # 1. Retrieve relevant course context.
        docs = self.retriever.invoke(question)
        context = self._format_context(docs)

        # 2. Ask Gemini for a structured recommendation grounded in context.
        messages = self.prompt.format_messages(
            context=context, history=self.history, question=question
        )
        response: RecommendationResponse = self.structured_llm.invoke(messages)

        # 3. Attach ground-truth source metadata.
        response = self._attach_sources(response)

        # 4. Custom tool: compute total learning hours for the sequence.
        sequence = response.learning_sequence or [
            c.course_name for c in response.recommended_courses
        ]
        response.tool_output = calculate_total_learning_hours.invoke(
            {"course_names": sequence}
        )
        response.total_learning_hours = calculate_total_hours_raw(sequence)

        # 5. Update conversation history for the next turn.
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=response.reason))

        return response

    def reset_history(self):
        self.history = []
