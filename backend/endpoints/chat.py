from fastapi import APIRouter
from schemas.schema_chat import ChatRequest, ChatResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


router = APIRouter()
dataset = load_dataset("wikipedia", "20220301.en", split="train[:1000]")
documents = [item["text"] for item in dataset]
# model_name = "Qwen/Qwen2.5-3B-Instruct"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embed_model.encode(documents, normalize_embeddings=True)
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

dimension = document_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(document_embeddings)

def get_context(query: str, k: int = 5):
    query_vector = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vector, k)
    return "\n".join([documents[i] for i in I[0]])


def generate_answer(query: str, mode: str, language: str, max_length: int = 256):
    prompt = f"Ты {mode} ассистент. Отвечай на вопрос на языке {language}:\n{get_context(query)}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_length)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    answer = generate_answer(request.query, request.mode, request.language)
    return ChatResponse(
        answer=answer,
        sources=[],
        confidence=1.0
    )
