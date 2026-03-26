import sys
import os
import logging
import glob
import asyncio
import threading
from pathlib import Path
import telegram
from telegram import Update
from flask import Flask, request
import chromadb
from groq import Groq
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========== НАСТРОЙКА ЛОГИРОВАНИЯ ==========
# На bothost.ru используем /tmp для логов, чтобы избежать проблем с правами доступа
LOG_DIR = Path("/tmp/bot_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "bot.log"

# Очищаем старые handlers чтобы избежать дублирования
root_logger = logging.getLogger()
root_logger.handlers = []

# Пишем логи в stderr (видно в консоли bothost) и в файл
handler_console = logging.StreamHandler(sys.stderr)
handler_console.setLevel(logging.DEBUG)
handler_console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

handler_file = logging.FileHandler(LOG_FILE, encoding='utf-8', mode='a')
handler_file.setLevel(logging.DEBUG)
handler_file.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[handler_console, handler_file]
)
logger = logging.getLogger(__name__)

# Принудительный вывод первого сообщения
logger.info("🚀 БОТ ЗАПУСКАЕТСЯ...")
sys.stderr.flush()

# ========== ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ==========
# На bothost.ru переменные задаются в панели управления
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DOMAIN = os.getenv("DOMAIN")
PORT = int(os.getenv("PORT", "3000"))

logger.info(f"🔍 Проверка переменных окружения:")
logger.info(f"   TELEGRAM_TOKEN: {'задан' if TELEGRAM_TOKEN else '❌ НЕ ЗАДАН'}")
logger.info(f"   GROQ_API_KEY: {'задан' if GROQ_API_KEY else '❌ НЕ ЗАДАН'}")
logger.info(f"   DOMAIN: {DOMAIN or '❌ НЕ ЗАДАН'}")
logger.info(f"   PORT: {PORT}")
sys.stderr.flush()

# Проверки обязательных переменных
if not TELEGRAM_TOKEN:
    logger.error("❌ TELEGRAM_TOKEN не задан в панели управления bothost.ru")
    sys.exit(1)
if not GROQ_API_KEY:
    logger.error("❌ GROQ_API_KEY не задан в панели управления bothost.ru")
    sys.exit(1)
if not DOMAIN:
    logger.error("❌ DOMAIN не задан в панели управления bothost.ru")
    sys.exit(1)

logger.info(f"✅ Токены получены: TELEGRAM_TOKEN={TELEGRAM_TOKEN[:5]}..., GROQ_API_KEY={GROQ_API_KEY[:5]}...")
logger.info(f"🌐 Домен: {DOMAIN}, порт: {PORT}")
sys.stderr.flush()

# ========== ИНИЦИАЛИЗАЦИЯ КЛИЕНТОВ ==========
# Groq
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq client создан")
    sys.stderr.flush()
except Exception as e:
    logger.error(f"❌ Ошибка создания Groq client: {e}")
    sys.exit(1)

# ChromaDB – используем относительные пути для bothost.ru
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"
DOCS_DIR = BASE_DIR / "documents"
CHROMA_PATH.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

try:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    logger.info("✅ Chroma client создан")
    sys.stderr.flush()
except Exception as e:
    logger.error(f"❌ Ошибка создания Chroma client: {e}")
    sys.exit(1)

# Модель эмбеддингов (может занимать ~500 МБ, учитывайте память)
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("✅ Модель эмбеддингов загружена")
    sys.stderr.flush()
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели эмбеддингов: {e}")
    sys.exit(1)

# Сплиттер текста
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)
TOP_K = 5

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С HTML ==========
def extract_text_and_links(html_path):
    """Извлекает текст и ссылки из HTML-файла"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            for tag in soup(['script', 'style', 'comment']):
                tag.decompose()
            links = [a['href'] for a in soup.find_all('a', href=True)]
            text = soup.get_text(separator='\n')
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
            return text, list(set(links))
    except Exception as e:
        logger.error(f"Ошибка чтения файла {html_path}: {e}")
        return "", []

def ensure_collection():
    """Создаёт коллекцию ChromaDB и индексирует HTML-файлы из DOCS_DIR"""
    logger.info("Проверяем коллекцию...")
    sys.stderr.flush()
    collection = None
    try:
        collection = chroma_client.get_collection("docs")
        logger.info("Коллекция уже существует")
        sys.stderr.flush()
        return collection
    except Exception as e:
        logger.info(f"Коллекция не найдена ({type(e).__name__}), создаём новую")
        sys.stderr.flush()
        collection = chroma_client.create_collection("docs")

        html_files = glob.glob(str(DOCS_DIR / "*.html"))
        if not html_files:
            logger.warning(f"Папка {DOCS_DIR} пуста или не содержит HTML-файлов. Индексация не выполнена.")
            sys.stderr.flush()
            return collection

        logger.info(f"Найдено {len(html_files)} HTML-файлов. Начинаем индексацию...")
        sys.stderr.flush()
        for file_path in html_files:
            logger.info(f"Индексируем {file_path}...")
            sys.stderr.flush()
            text, links = extract_text_and_links(file_path)
            if not text:
                logger.warning(f"  В файле {file_path} не удалось извлечь текст.")
                sys.stderr.flush()
                continue
            chunks = text_splitter.split_text(text)
            if not chunks:
                logger.warning(f"  В файле {file_path} нет текста для чанков.")
                sys.stderr.flush()
                continue

            embeddings = embedder.encode(chunks).tolist()
            file_name = Path(file_path).name
            ids = [f"{file_name}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": file_name, "links": ";".join(links)} for _ in chunks]

            collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"  Добавлено {len(chunks)} чанков, ссылок: {len(links)}")
            sys.stderr.flush()
        return collection

# Создаём коллекцию при старте
collection = ensure_collection()

# Создаём объект бота
bot = telegram.Bot(token=TELEGRAM_TOKEN)
logger.info("✅ Telegram bot создан")
sys.stderr.flush()

# ========== АСИНХРОННАЯ ОБРАБОТКА СООБЩЕНИЙ ==========
async def handle_update(update):
    """Обрабатывает входящее сообщение"""
    if not update.message:
        return
    user_text = update.message.text
    chat_id = update.message.chat_id
    logger.info(f"Пользователь {update.effective_user.id}: {user_text}")
    sys.stderr.flush()

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

# ========== НАСТРОЙКА ВЕБХУКА ==========
async def set_webhook():
    """Устанавливает вебхук для Telegram"""
    webhook_url = f"https://{DOMAIN}/webhook"
    try:
        await bot.set_webhook(webhook_url)
        logger.info(f"✅ Вебхук установлен: {webhook_url}")
        sys.stderr.flush()
    except Exception as e:
        logger.error(f"❌ Ошибка установки вебхука: {e}")
        sys.exit(1)

# ========== FLASK ПРИЛОЖЕНИЕ ==========
app = Flask(__name__)

# Глобальный цикл событий для асинхронных задач
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Точка входа для вебхука Telegram"""
    update = Update.de_json(request.get_json(force=True), bot)
    # Запускаем асинхронную обработку в фоновом цикле
    asyncio.run_coroutine_threadsafe(handle_update(update), loop)
    return 'ok'

def start_async_loop():
    """Запускает асинхронный цикл в отдельном потоке"""
    asyncio.set_event_loop(loop)
    loop.run_forever()

if __name__ == "__main__":
    # Запускаем фоновый поток с циклом событий
    threading.Thread(target=start_async_loop, daemon=True).start()

    # Устанавливаем вебхук (синхронно, но внутри вызывается await)
    asyncio.run(set_webhook())

    # Запускаем Flask
    logger.info(f"🚀 Запуск Flask на порту {PORT}")
    sys.stderr.flush()
    app.run(host="0.0.0.0", port=PORT)
