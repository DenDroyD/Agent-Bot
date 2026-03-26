import os
import logging
from dotenv import load_dotenv
import telegram
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# Инициализация векторной БД и эмбеддера
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="docs")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

TOP_K = 5  # количество релевантных чанков

async def start(update, context):
    await update.message.reply_text("Привет! Я бот-помощник по документации. Задай вопрос, и я найду ответ в нашей базе знаний.")

async def handle_message(update, context):
    user_text = update.message.text
    logger.info(f"Пользователь {update.effective_user.id}: {user_text}")

    # 1. Преобразуем вопрос в эмбеддинг
    query_embedding = embedder.encode([user_text]).tolist()[0]

    # 2. Ищем похожие чанки
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    if not results['documents'][0]:
        await update.message.reply_text("Извините, не нашёл информации по вашему вопросу.")
        return

    # 3. Собираем контекст из найденных чанков
    context_chunks = results['documents'][0]
    sources = []
    all_links = []
    for meta in results['metadatas'][0]:
        src = meta.get('source', 'неизвестно')
        sources.append(src)
        links = meta.get('links', '')
        if links:
            all_links.extend(links.split(';'))

    context_text = "\n\n".join(context_chunks)

    # 4. Формируем промпт
    system_prompt = (
        "Ты — полезный помощник, который отвечает на вопросы, используя только предоставленный контекст. "
        "Если ответа нет в контексте, скажи, что не знаешь. Не добавляй информацию из своего знания.\n\n"
        "Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:"
    )
    prompt = system_prompt.format(context=context_text, question=user_text)

    # 5. Вызываем Groq
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # можно заменить на другую модель
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка Groq: {e}")
        await update.message.reply_text("Произошла ошибка при генерации ответа. Попробуйте позже.")
        return

    # 6. Добавляем источники (имена файлов)
    source_line = f"\n\n📄 Источники: {', '.join(set(sources))}"
    final_answer = answer + source_line

    # (Опционально) можно добавить ссылки, но мы пока не выводим их, а только используем для индексации
    # Если нужно показать ссылки, раскомментируйте:
    # if all_links:
    #     final_answer += f"\n\n🔗 Ссылки: {', '.join(set(all_links))}"

    await update.message.reply_text(final_answer)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен в режиме long polling")
    app.run_polling(allowed_updates=telegram.Update.ALL_TYPES)

if __name__ == "__main__":
    main()