import os
import glob
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Конфигурация
DOCS_DIR = "documents"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "docs"

# Инициализация
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Модель для эмбеддингов (можно использовать и другую, например, all-MiniLM-L6-v2)
model = SentenceTransformer('all-MiniLM-L6-v2')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

def extract_text_from_html(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        # Удаляем скрипты, стили, комментарии
        for tag in soup(['script', 'style', 'comment']):
            tag.decompose()
        # Получаем текст
        text = soup.get_text(separator='\n')
        # Убираем лишние пустые строки
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)

def index_documents():
    # Очищаем коллекцию перед индексацией (опционально)
    # collection.delete(where={})   # раскомментируйте, если нужно переиндексировать

    for file_path in glob.glob(os.path.join(DOCS_DIR, "*.html")):
        print(f"Индексируем {file_path}...")
        text = extract_text_from_html(file_path)
        if not text:
            continue
        chunks = text_splitter.split_text(text)
        if not chunks:
            continue

        # Генерируем эмбеддинги для всех чанков
        embeddings = model.encode(chunks).tolist()

        # ID для каждого чанка (имя файла + номер)
        file_name = os.path.basename(file_path)
        ids = [f"{file_name}_{i}" for i in range(len(chunks))]

        # Метаданные (можно добавить больше)
        metadatas = [{"source": file_name} for _ in chunks]

        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"  Добавлено {len(chunks)} чанков.")

if __name__ == "__main__":
    index_documents()