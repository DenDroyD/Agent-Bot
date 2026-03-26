import sys
import os
import time
import traceback

# Папка для логов
LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "startup.log")

def write_log(msg):
    with open(log_file, "a") as f:
        f.write(msg + "\n")
    print(msg, file=sys.stderr)

write_log("=== Бот запускается ===")
write_log(f"Python: {sys.version}")
write_log(f"Рабочая директория: {os.getcwd()}")
write_log(f"Файлы в корне: {os.listdir('.')}")

# Проверка наличия папки documents
if os.path.isdir("documents"):
    write_log(f"Папка documents существует, файлы: {os.listdir('documents')}")
else:
    write_log("⚠️ Папка documents не найдена")

# Импорт модулей
try:
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
    write_log("✅ Все модули импортированы")
except Exception as e:
    write_log(f"❌ Ошибка импорта: {e}")
    traceback.print_exc(file=open(log_file, "a"))
    write_log("Ждём 60 секунд...")
    time.sleep(60)
    sys.exit(1)

write_log("Загружаем переменные окружения...")
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DOMAIN = os.getenv("DOMAIN")
write_log(f"TELEGRAM_TOKEN: {'есть' if TELEGRAM_TOKEN else 'нет'}")
write_log(f"GROQ_API_KEY: {'есть' if GROQ_API_KEY else 'нет'}")
write_log(f"DOMAIN: {DOMAIN}")

if not TELEGRAM_TOKEN:
    write_log("❌ TELEGRAM_TOKEN не задан")
    time.sleep(60)
    sys.exit(1)
if not GROQ_API_KEY:
    write_log("❌ GROQ_API_KEY не задан")
    time.sleep(60)
    sys.exit(1)

# Инициализация клиентов
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    write_log("✅ Groq client создан")
except Exception as e:
    write_log(f"❌ Groq client: {e}")
    traceback.print_exc(file=open(log_file, "a"))
    time.sleep(60)
    sys.exit(1)

try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    write_log("✅ Chroma client создан")
except Exception as e:
    write_log(f"❌ Chroma client: {e}")
    traceback.print_exc(file=open(log_file, "a"))
    time.sleep(60)
    sys.exit(1)

try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    write_log("✅ Модель эмбеддингов загружена")
except Exception as e:
    write_log(f"❌ Модель эмбеддингов: {e}")
    traceback.print_exc(file=open(log_file, "a"))
    time.sleep(60)
    sys.exit(1)

write_log("Все инициализации прошли успешно. Запускаем Flask...")

# Минимальный Flask для проверки
app = Flask(__name__)

@app.route('/')
def home():
    return "OK"

port = int(os.getenv("PORT", 3000))
write_log(f"Порт: {port}")
app.run(host="0.0.0.0", port=port)