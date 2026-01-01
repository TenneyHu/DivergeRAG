summary_prompt = """
You are given a question and multiple existing answers.

Question:
{QUESTION}

Existing answers:
{ANSWERS}

Task:
Identify the DISTINCT underlying views already present across the answers.

Guidelines:
- A "view" refers to a perspective, framing, or stance — not wording.
- Group answers that express the same core idea into one view.
- If two answers differ only in phrasing, treat them as the same view.
- Do NOT invent or infer new views.

Output requirements:
- Output a LIST of views.
- Each view must be a STRUCTURED ITEM with:
  - label: 2-5 words
  - description: exactly ONE sentence
- Keep the list concise and non-redundant.

Output format (strict):
Return ONLY a valid JSON array.
Do NOT include explanations, comments, or markdown.
[
  {{
    "label": "...",
    "description": "..."
  }},
  {{
    "label": "...",
    "description": "..."
  }}
]
"""

reflection_prompt = """
You are given an open-ended question and a list of views that have already been identified.

Question:
{QUESTION}

Existing views:
{VIEWS}

Task:
Reflect on the coverage of the existing views and identify ONE new, meaningful direction that is currently missing.

Guidelines:
- The new direction must be conceptually distinct from the existing views.
- Do NOT rephrase or refine any existing view.
- Avoid minor variations or subpoints.
- Focus on a different perspective, stakeholder, value system, time horizon, or level of abstraction.
- Do NOT generate a full answer.

Output requirements:
- Output exactly ONE new view.
- Use the structured format below.
- Be concise and precise.

New view format (STRICT):
{{
  "label": "...",          # 2-5 words summarizing the new direction
  "description": "..."    # exactly ONE sentence explaining the new direction
}}
"""

rag_prompt = """
Given the following context, identify the relevant information and generate a factually grounded, precise, and non-verbose answer.

Context:
{context_str}

Question:
{query_str}

Answer:
"""

rag_prompt_new_view = """
Given the following context, identify the relevant information and generate a factually grounded, precise, and non-verbose answer.
Context:
{context_str}

Question:
{query_str}

New view to emphasize:
- {view_label}: {view_description}

Instructions:
- Answer the original question.
- Emphasize insights relevant to the new view.

Answer:
"""

conditioned_query_prompt = """
You are generating a search query for retrieval.

Original query:
{QUESTION}

New view to explore:
- Label: {VIEW_LABEL}
- Description: {VIEW_DESCRIPTION}

Task:
Generate ONE refined search query that:
- Remains close to the original query
- Explicitly reflects the new view
- Is suitable for document retrieval (not a full sentence answer)

Guidelines:
- Reuse key terms from the original query where appropriate
- Add or replace terms to highlight the new view
- Keep the query concise
- Avoid questions, explanations, or full sentences

Output:
"""