import os
import json
from groq import Groq
try:
    from google import genai as google_genai
    GENAI_OK = True
except ImportError:
    GENAI_OK = False

def build_prompt(query, context_chunks, history):
    context = "\n\n---\n\n".join(f"[Page {c['page']}] {c['text']}" for c in context_chunks)
    hist_str = ""
    if history:
        lines = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in history[-6:]]
        hist_str = "\n\nConversation so far:\n" + "\n".join(lines) + "\n"

    system_message = """You are Folio, an expert document analysis assistant. You help users deeply understand their uploaded documents by providing thorough, insightful, and well-structured answers.

Your core principles:
1. **Be thorough**: Provide detailed, comprehensive answers. Don't be terse — explain concepts fully and give rich context from the document.
2. **Synthesize across pages**: When answering broad questions (summaries, themes, overviews), combine information from ALL provided context chunks to form a complete picture.
3. **Cite sources**: Reference page numbers naturally, e.g., "As discussed on page 5..." or "The author mentions on pages 3 and 7..."
4. **Handle different question types**:
   - For "what is this about" / summary questions → Provide a comprehensive overview covering the document's purpose, structure, key topics, and target audience.
   - For factual questions (who, what, when) → Give precise answers with page citations.
   - For structural questions (conventions, format, organization) → Describe the patterns and structure you observe in the content.
   - For opinion/analysis questions → Synthesize insights from the document context.
5. **Use formatting**: Use bullet points, numbered lists, and bold text to make answers scannable and organized.
6. **Never hallucinate**: Only state what's supported by the provided document context. If the context doesn't contain enough information, say so honestly but also share what you CAN infer from the available context."""

    user_message = f"""{hist_str}
Document context:
{context}

Question: {query}
Answer:"""

    return system_message, user_message

async def stream_groq(system_msg, user_msg):
    groq_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=groq_key)
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=2048,
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

async def stream_gemini(system_msg, user_msg):
    gemini_key = os.getenv("gemini") or os.getenv("GEMINI")
    client = google_genai.Client(api_key=gemini_key)
    prompt = f"{system_msg}\n\n{user_msg}"
    response = client.models.generate_content_stream(model="gemini-2.0-flash", contents=prompt)
    for chunk in response:
        yield chunk.text

async def generate_response_stream(query, context_chunks, history):
    system_msg, user_msg = build_prompt(query, context_chunks, history)
    
    success = False
    
    if os.getenv("GROQ_API_KEY"):
        try:
            async for chunk in stream_groq(system_msg, user_msg):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            success = True
        except Exception as e:
            print(f"Groq API Error: {e}")
            
    if not success and GENAI_OK and (os.getenv("gemini") or os.getenv("GEMINI")):
        try:
            async for chunk in stream_gemini(system_msg, user_msg):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            success = True
        except Exception as e:
            print(f"Gemini API Error: {e}")
            
    if not success:
        yield f"data: {json.dumps({'error': 'No API key found or APIs failed. Ensure GROQ_API_KEY or GEMINI key is valid.'})}\n\n"
        
    sources_data = [{"page": c["page"], "text": c["text"][:50] + "..."} for c in context_chunks]
    yield f"data: {json.dumps({'sources': sources_data, 'done': True})}\n\n"
