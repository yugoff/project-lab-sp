import json
from pathlib import Path

from fastapi import APIRouter
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss

from schemas.schema_chat import ChatRequest, ChatResponse


BASE_DIR = Path(__file__).resolve().parents[2]
router = APIRouter()
index = faiss.read_index(str(BASE_DIR / "scripts" / "full_directory" / "vector.index"))
with open(BASE_DIR / "scripts" / "full_directory" / "meta_data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

model_emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
model_tg = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_tg)
model = AutoModelForCausalLM.from_pretrained(model_tg, device_map="cpu")


def get_id_in_bd(query: str, k: int = 2):
    query_emb_np = model_emb.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype("float32")
    scores, ids = index.search(query_emb_np, k)

    return ids


def get_path(ids):
    with open(BASE_DIR / "scripts" / "full_directory" / "meta_data.json", "r", encoding="utf-8") as f:
        path = json.load(f)

        return path[ids[0][0]]["path"]


def get_context(ids):
    with open(BASE_DIR / "scripts" / "full_directory" / "meta_data.json", "r", encoding="utf-8") as f:
        context = json.load(f)

        return context[ids[0][0]]["text"]


def generate_answer(query: str, role: str, language: str = "ru", max_length: int = 512):
    id = get_id_in_bd(query)
    path = get_path(id)
    context = get_context(id)
    context_text = "\n".join(context) if context else "Информация отсутствует."

    prompt = (
        f"Ты корпоративный {role} ассистент.\n"
        f"Дай краткую сводку по контексту, ответь на вопрос.\n"
        f"Контекст:\n{context_text}\n\n"
        f"Вопрос: {query}\n"
        f"Ответ на языке только {language}:"
        f"Обязательно пропиши путь к файлу: {path}"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
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
