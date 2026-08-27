# CDLP Prompt Construction & System/User Roles Documentation

## 1. System vs. User Roles Overview
In modern LLM applications (especially RAG pipelines like **CDLP / ShipRule**), prompt design relies on strict role separation between system and user messages:

- **System Role (`role: "system"`)**: Serves as the control panel for model governance. It establishes the assistant's persona, domain boundaries, response format constraints, tone, and refusal rules for out-of-scope queries.
- **User Role (`role: "user"`)**: Represents the dynamic input per turn (e.g., an operational staff member asking for customs duty rates or import document requirements).

---

## 2. System Message Architecture (CDLP Constrained Prompt v2)

The production system prompt [`system_prompt_v2_constrained.txt`](file:///c:/Users/dhars/OneDrive/Desktop/ShipRule/prompts/system_prompt_v2_constrained.txt) enforces four core pillars:

1. **Role Definition**: Establishes identity as an official AI Support Assistant for the Customs Duty & Documentation Lookup Platform (CDLP).
2. **Strict Domain Scope**: Limits allowed queries strictly to logistics, customs duties, import documents, HS codes, restriction status, source agencies, and shipment regulations.
3. **Negative Scope & Refusal Rule**: Explicitly forbids answering non-logistics topics (cinema, movies, eating/food, sports, general trivia). Enforces a clear refusal fallback message.
4. **Format & Fallback Constraints**: Restricts responses to 2–3 sentences max unless a specific format (e.g., JSON) is requested, and provides a clear fallback when data is unmapped.

---

## 3. Comparison of Prompt Variations (Vague vs. Constrained)

| Feature | Prompt v1 (Vague System Prompt) | Prompt v2 (Constrained CDLP System Prompt) |
| :--- | :--- | :--- |
| **System Content** | *"You are a helpful assistant."* | Comprehensive role, scope, constraints, and refusal rules. |
| **Domain Guardrails** | None — answers any general topic (cinema, food, sports). | Strict — refuses non-logistics/customs topics instantly. |
| **Response Length** | Unpredictable (rambling, verbose, or conversational). | Tight (2–3 sentences maximum). |
| **Out-of-Scope Behavior** | Hallucinates or discusses movies, eating, etc. | Refuses with standardized CDLP fallback notice. |
| **Format Compliance** | Weak — may add conversational filler around requested JSON. | Strong — returns clean, parseable structured output. |

---

## 4. Why Prompt v2 Works Better (Justification)

1. **Eliminates Hallucination & Domain Drift**: By setting strict boundaries, staff cannot misuse the RAG assistant as a general chatbot for unrelated tasks like cinema or meal recommendations.
2. **Reliable Formatting for Parsing**: When system components or APIs parse model output (such as structured JSON `{ "duty_rate": ..., "required_docs": [...] }`), Prompt v2 reliably strips conversational intro/outro text.
3. **Auditability & Source Safety**: Grounding and refusal rules ensure that operational staff receive concise, source-traceable customs answers without hallucinated regulatory advice.

---

## 5. Constraining Model Output Format

To constrain the model to a specific format (e.g., strict JSON):
1. **Specify Schema in System/User Prompt**: State exact keys required: `"Reply ONLY with a valid JSON object matching the keys: {\"duty_rate\": string, \"required_documents\": list, \"restricted_status\": string}"`.
2. **Use Negative Directives**: Add `"Do not include markdown formatting, preambles, or explanations outside the JSON object."`
3. **API-level Enforcement**: Set `response_format={"type": "json_object"}` in model API calls where supported.
