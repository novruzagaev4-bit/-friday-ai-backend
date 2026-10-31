
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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "name": "F.R.I.D.A.Y.",
        "status": "online",
        "provider": "OpenRouter",
        "model": OPENROUTER_MODEL
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "provider": "OpenRouter"
    }


@app.post("/chat")
def chat(data: ChatRequest):
    message = data.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Сообщение пустое"
        )

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY не настроен"
        )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты F.R.I.D.A.Y. — персональный "
                    "ИИ-помощник. Отвечай на русском языке, "
                    "конкретно, грамотно и честно. "
                    "Помогай с программированием, технологиями "
                    "и развитием проекта QADAM AI. "
                    "Сохраняй дружелюбный и профессиональный стиль."
                )
            },
            {
                "role": "user",
                "content": message
            }
        ],
        "max_tokens": 500
    }

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": "https://friday-ai-interface.onrender.com",
            "X-OpenRouter-Title": "F.R.I.D.A.Y. AI"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=90
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        choices = result.get("choices", [])

        if not choices:
            raise HTTPException(
                status_code=502,
                detail="OpenRouter не вернул ответ модели."
            )

        answer = choices[0].get(
            "message", {}
        ).get("content")

        if isinstance(answer, list):
            answer = "".join(
                item.get("text", "")
                for item in answer
                if isinstance(item, dict)
            )

        if not answer:
            answer = "Модель не вернула текстовый ответ."

        return {
            "reply": answer,
            "provider": "OpenRouter",
            "model": result.get("model", OPENROUTER_MODEL)
        }

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(
            "utf-8", errors="replace"
        )

        print(
            "OpenRouter HTTP error:",
            exc.code,
            error_body[:1500]
        )

        if exc.code == 401:
            detail = "Неверный или недействительный API-ключ."
        elif exc.code == 402:
            detail = "Недостаточно доступного баланса или квоты."
        elif exc.code == 403:
            detail = "OpenRouter отклонил запрос. Проверь доступ."
        elif exc.code == 404:
            detail = "Запрошенная модель недоступна."
        elif exc.code == 429:
            detail = "Превышен лимит запросов. Попробуй позже."
        else:
            detail = f"OpenRouter вернул HTTP {exc.code}."

        raise HTTPException(
            status_code=502,
            detail=detail
        ) from exc

    except urllib.error.URLError as exc:
        print("OpenRouter connection error:", str(exc.reason))

        raise HTTPException(
            status_code=502,
            detail="Не удалось подключиться к OpenRouter."
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        print("F.R.I.D.A.Y. error:", repr(exc))

        raise HTTPException(
            status_code=502,
            detail="Внутренняя ошибка обработки ответа."
        ) from exc
