import os
import glob
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Путь к папке с документами
DOCS_DIR = "documents"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "docs"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

model = SentenceTransformer('all-MiniLM-L6-v2')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

def extract_text_and_links(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        for tag in soup(['script', 'style', 'comment']):
            tag.decompose()
        links = []
        for a in soup.find_all('a', href=True):
            links.append(a['href'])
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        return text, list(set(links))

def index_documents():
    # Если нужно переиндексировать с нуля, раскомментируйте:
    # collection.delete(where={})
    
    for file_path in glob.glob(os.path.join(DOCS_DIR, "*.html")):
        print(f"Индексируем {file_path}...")
        text, links = extract_text_and_links(file_path)
        if not text:
            continue
        chunks = text_splitter.split_text(text)
        if not chunks:
            continue

        embeddings = model.encode(chunks).tolist()
        file_name = os.path.basename(file_path)
        ids = [f"{file_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_name, "links": ";".join(links)} for _ in chunks]

        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"  Добавлено {len(chunks)} чанков, ссылок: {len(links)}")

if __name__ == "__main__":
    index_documents()