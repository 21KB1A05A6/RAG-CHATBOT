# Course Recommendation Assistant

A RAG (Retrieval-Augmented Generation) application built with **LangChain** and
**Google Gemini** that recommends courses from a small catalog based on a
learner's background and goal, e.g.:

> "I am an SAP ABAP developer with no AI experience. Which course should I
> take first to learn SAP Business AI?"

## Project layout

```
course_recommendation_assistant/
├── courses_data.py   # 6 sample course documents (catalog + LangChain Documents)
├── models.py          # Pydantic models for the structured output
├── tools.py            # Custom tool: calculate_total_learning_hours
├── rag_engine.py        # Retrieval + Gemini + structured output + history
├── app.py                # Streamlit front end
├── requirements.txt
└── .env.example
```

## How it works

1. **Course catalog** (`courses_data.py`): 6 sample courses, each with
   `course_name`, `skills`, `experience_level`, `duration_hours`, and
   `prerequisites`. Each course is also rendered into a text `Document` for
   embedding.
2. **Retrieval**: course documents are embedded with
   `GoogleGenerativeAIEmbeddings` (`models/embedding-001`) into a **Chroma**
   vector store. The top-k most relevant courses are retrieved for each
   question.
3. **Structured generation**: `ChatGoogleGenerativeAI` (`gemini-2.0-flash`)
   is used with `.with_structured_output(RecommendationResponse)` so Gemini's
   answer is *forced* into the `RecommendationResponse` Pydantic schema —
   `recommended_courses`, `reason`, `prerequisites`, `learning_sequence`,
   `confidence`.
4. **Source metadata**: after generation, each recommended course's metadata
   (`course_id`, `source`, `experience_level`, `duration`) is re-attached
   directly from the catalog, so it's always accurate (not hallucinated).
5. **Custom tool** (`tools.py`): `calculate_total_learning_hours` is a
   proper `@tool`-decorated LangChain tool that sums course durations for a
   given `learning_sequence`. It's invoked automatically after every answer
   and the result is attached to the response (`total_learning_hours`,
   `tool_output`).
6. **Conversation history**: the assistant keeps a running list of
   `HumanMessage` / `AIMessage` objects and feeds it back into the prompt on
   every turn, so follow-up questions ("what about after that?") have
   context.
7. **Streamlit UI** (`app.py`): chat interface, sidebar catalog browser, and
   a formatted rendering of the structured response (course cards,
   prerequisites, learning path, confidence, total hours).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Get a Google API key from https://aistudio.google.com/app/apikey
3. Either export it as an environment variable:
   ```bash
   export GOOGLE_API_KEY=your_key_here
   ```
   or paste it directly into the Streamlit sidebar at runtime.
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Using it without Streamlit (Python / script)

```python
from rag_engine import CourseRecommendationAssistant

assistant = CourseRecommendationAssistant(google_api_key="YOUR_KEY")

result = assistant.ask(
    "I am an SAP ABAP developer with no AI experience. "
    "Which course should I take first to learn SAP Business AI?"
)

print(result.model_dump_json(indent=2))

# Follow-up question uses conversation history automatically
followup = assistant.ask("How many total hours will that take me?")
print(followup.total_learning_hours)
```

`result` is a `RecommendationResponse` Pydantic model, e.g.:

```json
{
  "recommended_courses": [
    {
      "course_name": "Introduction to Artificial Intelligence",
      "skills_covered": ["AI concepts", "Machine learning basics", "..."],
      "duration": "15 hours",
      "experience_level": "Beginner",
      "match_reason": "You have no prior AI exposure, so this builds the foundation before any SAP-specific AI course.",
      "source": {
        "course_id": "AI-INTRO-101",
        "course_name": "Introduction to Artificial Intelligence",
        "source": "course_catalog::AI-INTRO-101",
        "experience_level": "Beginner",
        "duration": "15 hours"
      }
    }
  ],
  "reason": "As an ABAP developer with no AI background, you should first build general AI literacy before tackling SAP-specific AI tooling.",
  "prerequisites": ["None"],
  "learning_sequence": [
    "Introduction to Artificial Intelligence",
    "SAP Business AI Foundations",
    "SAP BTP and AI Integration"
  ],
  "confidence": 0.9,
  "total_learning_hours": 85.0,
  "tool_output": "Total learning hours: 85 (Introduction to Artificial Intelligence: 15h; SAP Business AI Foundations: 20h; SAP BTP and AI Integration: 35h; ...)"
}
```

## Requirements checklist

| Requirement | Where |
|---|---|
| 5+ sample courses with name, skills, level, duration, prerequisites | `courses_data.py` |
| RAG application answering natural-language questions | `rag_engine.py` (Chroma retriever + Gemini) |
| Structured output: recommended_courses, reason, prerequisites, learning_sequence, confidence | `models.py` / `RecommendationResponse` |
| Custom tool for total learning hours | `tools.py` / `calculate_total_learning_hours` |
| Conversation history | `CourseRecommendationAssistant.history` in `rag_engine.py` |
| Pydantic model output | `models.py`, returned directly from `ask()` |
| Source metadata per recommendation | `SourceMetadata` in `models.py`, attached in `_attach_sources` |
| Streamlit front end | `app.py` |

## Notes

- The retrieval step uses semantic search, so the LLM never sees the full
  catalog — only the top-k most relevant courses — which keeps answers
  grounded and avoids hallucinated course names beyond the catalog.
- If a course name the model returns doesn't exactly match the catalog
  (e.g. paraphrased), `_attach_sources` will simply leave the placeholder
  source in place — in a production system you'd add fuzzy matching here.
- Swap `gemini-2.0-flash` for another Gemini model in `rag_engine.py` if you
  have access to a different one.
