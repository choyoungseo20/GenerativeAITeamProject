from fastapi import FastAPI, File, UploadFile, Depends
from sentence_transformers import SentenceTransformer
from io import BytesIO
from mimetypes import guess_type
from PIL import Image
from open_clip import tokenizer
import torch
import faiss
import spacy
import threading
import PyPDF2
import magic
import open_clip

# RAG Service 클래스 정의
class RAGService:
    def __init__(self):
        # 뮤텍스
        self.lock = threading.Lock()

        # spaCy 모델 로드
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp_kr = spacy.load("ko_core_news_lg")

        # OpenCLIP 모델 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
        self.model = self.model.to(self.device)
        self.text_descriptions = [
            "OpenAI",
            "Rocket",
            "Nobel Prize",
            "Google"
        ]

        # SentenceTransformer 모델 로드
        self.text_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        # FAISS 초기화
        init_embedding = self.text_model.encode(["Hello RAG"])
        dim = init_embedding.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.corpus = ["Hello RAG"]
        self.index.add(init_embedding)

    def indexing(self, text: str):
        with self.lock:  # 동기화 시작
            doc = self.nlp(text)
            corpus = [sent.text for sent in doc.sents]
            embeddings = self.text_model.encode(corpus)
            self.index.add(embeddings)
            self.corpus.extend(corpus)

    def retrieval(self, query: str):
        query_embedding = self.text_model.encode([query])
        k = 2
        D, I = self.index.search(query_embedding, k)
        return [self.corpus[idx] for idx in I[0]]

    def init_index(self):
        init_embedding = self.text_model.encode(["Hello RAG"])
        dim = init_embedding.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.corpus = ["Hello RAG"]
        self.index.add(init_embedding)

    # clip
    def clip_image(self, image):
        text_tokens = tokenizer.tokenize(self.text_descriptions).to(self.device)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 이미지 및 텍스트 특징 추출
            image_features = self.model.encode_image(image_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_features = self.model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # 유사도 계산 및 확률화
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # 최상위 5개의 확률과 인덱스 반환
        top_probs, top_indices = text_probs[0].topk(4)  # text_probs[0]: 1D 텐서로 변환
        return top_probs.tolist(), top_indices.tolist()  # 리스트로 변환


# FastAPI 앱 설정
app = FastAPI()

# RAGService 인스턴스 생성
rag_service = RAGService()


@app.get("/")
async def root():
    return {"message": "Welcome to RAG API"}


@app.post("/indexing")
async def indexing_text(text: str, service: RAGService = Depends(lambda: rag_service)):
    service.indexing(text)

    return {"message": "Text Indexing complete"}

@app.post("/indexing/file")
async def index_file(file: UploadFile = File(...), service: RAGService = Depends(lambda: rag_service)):
    # 파일 확장자 및 MIME 타입 확인
    mime_type, _ = guess_type(file.filename)
    detected_mime_type = magic.from_buffer(await file.read(1024), mime=True)  # 파일 내용으로 MIME 타입 확인

    # PDF 파일 처리
    if mime_type == "application/pdf" or detected_mime_type == "application/pdf":
        # 파일 내용을 다시 읽어야 하므로 스트림 위치를 리셋
        file.file.seek(0)
        pdf_reader = PyPDF2.PdfReader(BytesIO(await file.read()))
        pdf_text = ""

        for page in pdf_reader.pages:
            pdf_text += page.extract_text()  # 각 페이지의 텍스트를 추가

        service.indexing(pdf_text)

        return {"message": "PDF Indexing complete"}

    # 이미지 파일 처리
    elif mime_type and mime_type.startswith("image/") or detected_mime_type.startswith("image/"):
        # 파일 내용을 다시 읽어야 하므로 스트림 위치를 리셋
        file.file.seek(0)
        image_reader = Image.open(BytesIO(await file.read()))

        top_probs, top_indices = service.clip_image(image_reader)

        max_prob_idx = top_indices[0]  # 첫 번째 인덱스 사용
        best_description = service.text_descriptions[max_prob_idx]

        service.indexing(best_description)

        return {"message": "Image Indexing complete"}

    # 기타 파일 처리
    else:
        return {"error": f"Unsupported file type: {detected_mime_type}"}



@app.get("/retrieval")
async def retrieval_text(query: str, service: RAGService = Depends(lambda: rag_service)):
    result = service.retrieval(query)

    return {"retrieved_chunks": result}

@app.post("/retrieval/file")
async def retrieval_file(query: str, file: UploadFile = File(...), service: RAGService = Depends(lambda: rag_service)):
    image_reader = Image.open(BytesIO(await file.read()))

    top_probs, top_indices = service.clip_image(image_reader)

    max_prob_idx = top_indices[0]  # 첫 번째 인덱스 사용
    best_description = service.text_descriptions[max_prob_idx]

    result = service.retrieval(best_description + query)

    return {"retrieved_chunks": result}

@app.post("/init/index")
async def init_index(service: RAGService = Depends(lambda: rag_service)):
    service.init_index()

    return {"message": "Init complete"}

@app.post("/clip/test")
async def clip_test(file: UploadFile = File(...), service: RAGService = Depends(lambda: rag_service)):
    # 업로드된 파일을 읽고 이미지로 변환
    image_reader = Image.open(BytesIO(await file.read()))

    # clip_image 메서드 호출
    top_probs, top_indices = service.clip_image(image_reader)

    # 최상위 확률에 해당하는 텍스트 설명 가져오기
    max_prob_idx = top_indices[0]  # 첫 번째 인덱스 사용
    best_description = service.text_descriptions[max_prob_idx]

    return {"best_description": best_description, "top_probs": top_probs, "top_indices": top_indices}