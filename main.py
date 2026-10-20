
import os
import json
import urllib.request
import urllib.error

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="F.R.I.D.A.Y. API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://friday-ai-interface.onrender.com"
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "name": "F.R.I.D.A.Y.",
        "status": "online",
        "provider": "Groq"
    }


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

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="На сервере не настроен GROQ_API_KEY"
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты F.R.I.D.A.Y., персональный "
                    "ИИ-помощник. Отвечай на русском языке, "
                    "конкретно и честно. Помогай развивать "
                    "проект QADAM AI."
                )
            },
            {
                "role": "user",
                "content": message
            }
        ],
        "max_tokens": 300
    }

    request = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "FRIDAY-AI/1.0"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=45
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        choices = result.get("choices", [])

        if not choices:
            raise HTTPException(
                status_code=502,
                detail="Groq вернул пустой список ответов."
            )

        answer = choices[0].get(
            "message", {}
        ).get("content")

        if not answer:
            answer = "Модель не вернула текстовый ответ."

        return {"reply": answer}

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(
            "utf-8", errors="replace"
        )

        print(
            "Groq HTTP error:",
            exc.code,
            error_body[:1000]
        )

        if exc.code == 401:
            detail = "Groq отклонил API-ключ. Проверь GROQ_API_KEY."
        elif exc.code == 403:
            detail = (
                "Groq или его защитная система заблокировали "
                "запрос. Проверь Render Logs."
            )
        elif exc.code == 429:
            detail = (
                "Превышен лимит запросов Groq. "
                "Попробуй позже."
            )
        else:
            detail = f"Groq вернул HTTP {exc.code}."

        raise HTTPException(
            status_code=502,
            detail=detail
        ) from exc

    except urllib.error.URLError as exc:
        print("Groq connection error:", str(exc.reason))
        raise HTTPException(
            status_code=502,
            detail="Не удалось подключиться к Groq."
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        print("Groq model error:", repr(exc))
        raise HTTPException(
            status_code=502,
            detail="Внутренняя ошибка обработки ответа модели."
        ) from exc
