import sys
import os
import logging
import glob
import traceback
import asyncio
import concurrent.futures
from dotenv import load_dotenv
import telegram
from flask import Flask, request
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Папка для логов
LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(os.path.join(LOG_DIR, "bot.log"))
    ]
)
logger = logging.getLogger(__name__)

logger.info("🚀 Бот запускается...")

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN:
    logger.error("❌ TELEGRAM_TOKEN не задан")
    sys.exit(1)
if not GROQ_API_KEY:
    logger.error("❌ GROQ_API_KEY не задан")
    sys.exit(1)

logger.info(f"✅ Токены получены: TELEGRAM_TOKEN={TELEGRAM_TOKEN[:5]}..., GROQ_API_KEY={GROQ_API_KEY[:5]}...")

# Инициализация Groq
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq client создан")
except Exception as e:
    logger.error(f"❌ Ошибка создания Groq client: {e}")
    sys.exit(1)

# Конфигурация векторной БД
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "docs"
DOCS_DIR = "documents"

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    logger.info("✅ Chroma client создан")
except Exception as e:
    logger.error(f"❌ Ошибка создания Chroma client: {e}")
    sys.exit(1)

# Модель эмбеддингов
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("✅ Модель эмбеддингов загружена")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели эмбеддингов: {e}")
    sys.exit(1)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

TOP_K = 5

def extract_text_and_links(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        for tag in soup(['script', 'style', 'comment']):
            tag.decompose()
        links = [a['href'] for a in soup.find_all('a', href=True)]
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        return text, list(set(links))

def ensure_collection():
    logger.info("Проверяем коллекцию...")
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        logger.info("Коллекция уже существует")
        return collection
    except chromadb.errors.InvalidCollectionException:
        logger.info("Коллекция не найдена, создаём новую")
        collection = chroma_client.create_collection(COLLECTION_NAME)

        html_files = glob.glob(os.path.join(DOCS_DIR, "*.html"))
        if not html_files:
            logger.warning(f"Папка {DOCS_DIR} пуста или не содержит HTML-файлов. Индексация не выполнена.")
            return collection

        logger.info(f"Найдено {len(html_files)} HTML-файлов. Начинаем индексацию...")
        for file_path in html_files:
            logger.info(f"Индексируем {file_path}...")
            text, links = extract_text_and_links(file_path)
            if not text:
                logger.warning(f"  В файле {file_path} не удалось извлечь текст.")
                continue
            chunks = text_splitter.split_text(text)
            if not chunks:
                logger.warning(f"  В файле {file_path} нет текста для чанков.")
                continue

            embeddings = embedder.encode(chunks).tolist()
            file_name = os.path.basename(file_path)
            ids = [f"{file_name}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": file_name, "links": ";".join(links)} for _ in chunks]

            collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"  Добавлено {len(chunks)} чанков, ссылок: {len(links)}")
        return collection

# Инициализируем коллекцию один раз при старте
collection = ensure_collection()

# Создаём объект бота
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# --- Flask приложение для вебхука ---
app = Flask(__name__)

# Обработчик вебхука
@app.route('/webhook', methods=['POST'])
def webhook():
    # Получаем обновление от Telegram
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    # Запускаем обработку в отдельном потоке, чтобы не блокировать Flask
    def run_async():
        asyncio.run(handle_update(update))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(run_async)
    executor.shutdown(wait=False)
    return 'ok'

# Функция обработки обновления (асинхронная)
async def handle_update(update):
    if update.message:
        user_text = update.message.text
        chat_id = update.message.chat_id
        logger.info(f"Пользователь {update.effective_user.id}: {user_text}")

        # Поиск в коллекции
        query_embedding = embedder.encode([user_text]).tolist()[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
        )

        if not results['documents'][0]:
            await bot.send_message(chat_id, "Извините, не нашёл информации по вашему вопросу.")
            return

        context_chunks = results['documents'][0]
        sources = list(set(meta['source'] for meta in results['metadatas'][0]))
        context_text = "\n\n".join(context_chunks)

        system_prompt = (
            "Ты — полезный помощник, который отвечает на вопросы, используя только предоставленный контекст. "
            "Если ответа нет в контексте, скажи, что не знаешь. Не добавляй информацию из своего знания.\n\n"
            "Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:"
        )
        prompt = system_prompt.format(context=context_text, question=user_text)

        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            answer = completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Ошибка Groq: {e}")
            await bot.send_message(chat_id, "Произошла ошибка при генерации ответа. Попробуйте позже.")
            return

        source_line = f"\n\n📄 Источники: {', '.join(sources)}"
        final_answer = answer + source_line
        await bot.send_message(chat_id, final_answer)

# Установка вебхука
async def set_webhook():
    domain = os.getenv("DOMAIN")
    if not domain:
        logger.error("❌ Переменная DOMAIN не задана. Убедитесь, что домен включён в настройках бота.")
        sys.exit(1)
    webhook_url = f"https://{domain}/webhook"
    try:
        await bot.set_webhook(webhook_url)
        logger.info(f"✅ Вебхук установлен: {webhook_url}")
    except Exception as e:
        logger.error(f"❌ Ошибка установки вебхука: {e}")
        sys.exit(1)

# Запуск Flask
if __name__ == "__main__":
    # Устанавливаем вебхук (только при первом запуске)
    asyncio.run(set_webhook())

    # Запускаем Flask-сервер
    port = int(os.getenv("PORT", 3000))
    logger.info(f"🚀 Запуск Flask на порту {port}")
    app.run(host="0.0.0.0", port=port)