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
        lines = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in history[-4:]]
        hist_str = "\n\nConversation so far:\n" + "\n".join(lines) + "\n"
        
    return f"""You are an intelligent document assistant. Use the document context provided below to answer the user's question.
If the user asks for a summary or what the document is about, use the provided context to synthesize the best possible description. 
Be concise and cite page numbers when relevant. 
If the context contains absolutely no relevant information to answer a specific factual question, say: "I couldn't find that in the document."{hist_str}

Document context:
{context}

Question: {query}
Answer:"""

async def stream_groq(prompt):
    groq_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=groq_key)
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

async def stream_gemini(prompt):
    gemini_key = os.getenv("gemini") or os.getenv("GEMINI")
    client = google_genai.Client(api_key=gemini_key)
    response = client.models.generate_content_stream(model="gemini-2.0-flash", contents=prompt)
    for chunk in response:
        yield chunk.text

async def generate_response_stream(query, context_chunks, history):
    prompt = build_prompt(query, context_chunks, history)
    
    success = False
    
    if os.getenv("GROQ_API_KEY"):
        try:
            async for chunk in stream_groq(prompt):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            success = True
        except Exception as e:
            print(f"Groq API Error: {e}")
            
    if not success and GENAI_OK and (os.getenv("gemini") or os.getenv("GEMINI")):
        try:
            async for chunk in stream_gemini(prompt):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            success = True
        except Exception as e:
            print(f"Gemini API Error: {e}")
            
    if not success:
        yield f"data: {json.dumps({'error': 'No API key found or APIs failed. Ensure GROQ_API_KEY or GEMINI key is valid.'})}\n\n"
        
    sources_data = [{"page": c["page"], "text": c["text"][:50] + "..."} for c in context_chunks]
    yield f"data: {json.dumps({'sources': sources_data, 'done': True})}\n\n"
