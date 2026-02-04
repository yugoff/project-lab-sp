from fastapi import APIRouter
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

from schemas.schema_chat import ChatRequest, ChatResponse


router = APIRouter()
index = faiss.read_index("scripts/roles_index.faiss")
with open("scripts/dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

model_emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model_tg = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_tg)
model = AutoModelForCausalLM.from_pretrained(model_tg, device_map="auto")


def get_context(query: str, role: str, k: int = 2):
    query_emb = model_emb.encode([query], convert_to_tensor=True)
    query_emb_np = np.array(query_emb.cpu()).astype("float32")
    faiss.normalize_L2(query_emb_np)

    scores, ids = index.search(query_emb_np, k)
    results = []

    for idx, score in zip(ids[0], scores[0]):
        doc = dataset[idx]
        if doc["role"].lower() == role.lower():
            results.append(doc["text"])
        if len(results) >= k:
            break

    return results


def generate_answer(query: str, role: str, language: str = "ru", max_length: int = 36):
    context = get_context(query, role)
    context_text = "\n".join(context) if context else "Информация отсутствует."

    prompt = (
        f"Ты корпоративный {role} ассистент.\n"
        f"Используй только информацию из контекста ниже.\n"
        f"Контекст:\n{context_text}\n\n"
        f"Вопрос: {query}\n"
        f"Ответ на языке {language}:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return answer.strip()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    answer = generate_answer(request.query, request.mode, request.language)
    return ChatResponse(
        answer=answer,
        sources=[],
        confidence=1.0
    )
