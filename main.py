import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient

app = FastAPI(title="F.R.I.D.A.Y. API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://friday-ai-interface.onrender.com"
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

MODEL_ID = os.getenv(
    "MODEL_ID",
    "Qwen/Qwen3-4B-Instruct-2507"
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"name": "F.R.I.D.A.Y.", "status": "online"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(data: ChatRequest):
    message = data.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Сообщение пустое"
        )

    token = os.getenv("HF_TOKEN")
    if not token:
        raise HTTPException(
            status_code=503,
            detail="На сервере не настроен HF_TOKEN"
        )

    try:
        client = InferenceClient(
            model=MODEL_ID,
            token=token
        )

        result = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты F.R.I.D.A.Y., персональный "
                        "ИИ-помощник. Отвечай на русском, "
                        "конкретно и честно. Помогай "
                        "развивать проект QADAM AI."
                    )
                },
                {"role": "user", "content": message}
            ],
            max_tokens=200,
        )

        answer = result.choices[0].message.content
        return {"reply": answer or "Нет ответа от модели."}

    except Exception as exc:
        print("Model error:", repr(exc))
        raise HTTPException(
            status_code=502,
            detail="Не удалось получить ответ модели."
        )
