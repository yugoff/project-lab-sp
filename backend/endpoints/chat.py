from fastapi import APIRouter
from schemas.schema_chat import ChatRequest, ChatResponse
from transformers import AutoModelForCausalLM, AutoTokenizer


router = APIRouter()

model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def generate_answer(query: str, mode: str, language: str, max_length: int = 256):
    prompt = f"Ты {mode} ассистент. Отвечай на вопрос на языке {language}:\n{query}"
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
