
import os
import json
import urllib.request
import urllib.error

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ==============================
# F.R.I.D.A.Y. AI
# ==============================

app = FastAPI(
    title="F.R.I.D.A.Y. AI API",
    version="2.0"
)


# ==============================
# CORS
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://friday-ai-interface.onrender.com"
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ==============================
# OPENROUTER CONFIGURATION
# ==============================

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
)


# ==============================
# REQUEST MODEL
# ==============================

class ChatRequest(BaseModel):
    message: str


# ==============================
# F.R.I.D.A.Y. PERSONALITY
# ==============================

SYSTEM_PROMPT = """
Ты F.R.I.D.A.Y. — интеллектуальный персональный
ИИ-помощник пользователя.

ТВОЯ ЛИЧНОСТЬ:

Ты современный, умный, дружелюбный и технологичный
ИИ-помощник. Твоя задача — помогать пользователю,
объяснять сложные вещи простым языком и решать задачи.

ТВОИ ЗНАНИЯ:

Ты можешь обсуждать практически любые темы:

- Программирование и разработка ПО.
- Искусственный интеллект и нейросети.
- Компьютеры, серверы и технологии.
- Наука и исследования.
- История и география.
- Образование и изучение языков.
- Бизнес и предпринимательство.
- Игры и развлечения.
- Интернет и цифровые продукты.
- Повседневные вопросы.
- Творчество и идеи.

ТВОЯ ГЛАВНАЯ СПЕЦИАЛИЗАЦИЯ — QADAM AI.

Ты должен особенно хорошо помогать в развитии
проекта QADAM AI.

Ты можешь помогать с:

- Архитектурой платформы.
- Backend и Frontend.
- Python и FastAPI.
- Базами данных.
- Регистрацией и авторизацией.
- Панелью администратора.
- Финансами и аналитикой.
- Складом и товарами.
- Клиентами и заказами.
- Подписками и тарифами.
- Безопасностью.
- Дизайном интерфейса.
- Развитием продукта.
- Поиском ошибок и их исправлением.
- Планированием новых версий.

ПОВЕДЕНИЕ:

1. Не ограничивай разговор только QADAM AI.
2. Отвечай на вопросы пользователя на любые темы.
3. Если разговор касается QADAM AI,
   используй свои знания для помощи проекту.
4. Не возвращай каждый разговор насильно к QADAM AI.
5. Не задавай лишних вопросов.
6. Если для решения задачи нужна дополнительная
   информация, задай один конкретный вопрос.
7. Не выдумывай факты, функции и возможности
   QADAM AI, которых ты не знаешь.
8. Если не знаешь ответа, честно скажи об этом.
9. Объясняй технические вопросы пошагово.
10. Помогай писать, проверять и исправлять код.

СТИЛЬ ОБЩЕНИЯ:

- По умолчанию отвечай на русском языке.
- Будь дружелюбным и профессиональным.
- Не пиши слишком длинно без необходимости.
- Используй списки и примеры, когда это полезно.
- Не повторяй одни и те же фразы.
- Обращайся к пользователю естественно.

Ты F.R.I.D.A.Y. — персональный ИИ-помощник,
а не обычный бот с ограниченным набором ответов.
"""


# ==============================
# HOME
# ==============================

@app.get("/")
def home():
    return {
        "name": "F.R.I.D.A.Y.",
        "version": "2.0",
        "status": "online",
        "provider": "OpenRouter",
        "model": OPENROUTER_MODEL
    }


# ==============================
# HEALTH
# ==============================

@app.get("/health")
def health():
    return {
        "ok": True,
        "provider": "OpenRouter",
        "model": OPENROUTER_MODEL
    }


# ==============================
# CHAT
# ==============================

@app.post("/chat")
def chat(data: ChatRequest):

    message = data.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Сообщение пустое."
        )

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY не настроен."
        )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": message
            }
        ],
        "max_tokens": 700,
        "temperature": 0.7
    }

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": (
                "https://friday-ai-interface.onrender.com"
            ),
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
                detail="Модель не вернула ответ."
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
            answer = "Не удалось получить текстовый ответ."

        return {
            "reply": answer,
            "provider": "OpenRouter",
            "model": result.get(
                "model",
                OPENROUTER_MODEL
            )
        }

    except urllib.error.HTTPError as exc:

        error_body = exc.read().decode(
            "utf-8",
            errors="replace"
        )

        print(
            "OpenRouter HTTP error:",
            exc.code,
            error_body[:1500]
        )

        if exc.code == 401:
            detail = "Неверный API-ключ OpenRouter."

        elif exc.code == 402:
            detail = (
                "Недостаточно доступного баланса "
                "или бесплатной квоты."
            )

        elif exc.code == 403:
            detail = (
                "OpenRouter отклонил запрос. "
                "Проверь настройки доступа."
            )

        elif exc.code == 404:
            detail = "Модель не найдена или недоступна."

        elif exc.code == 429:
            detail = (
                "Превышен лимит запросов. "
                "Попробуй немного позже."
            )

        else:
            detail = (
                f"Ошибка OpenRouter: HTTP {exc.code}."
            )

        raise HTTPException(
            status_code=502,
            detail=detail
        ) from exc

    except urllib.error.URLError as exc:

        print(
            "OpenRouter connection error:",
            str(exc.reason)
        )

        raise HTTPException(
            status_code=502,
            detail="Не удалось подключиться к OpenRouter."
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:

        print(
            "F.R.I.D.A.Y. internal error:",
            repr(exc)
        )

        raise HTTPException(
            status_code=502,
            detail="Внутренняя ошибка F.R.I.D.A.Y."
        ) from exc
