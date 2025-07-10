# search_app.py

import streamlit as st
import numpy as np
import operator
import sklearn.metrics
import re
import os
from collections import defaultdict

# NLTK imports (Đảm bảo đã tải các gói này nếu chạy lần đầu trên môi trường mới)
import nltk
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

# Thư viện cho BM25 (cần cài đặt: pip install rank_bm25)
from rank_bm25 import BM25Okapi

# Thư viện cho Semantic Search (cần cài đặt: pip install sentence-transformers)
from sentence_transformers import SentenceTransformer

# --- 1. Cấu hình và Tải dữ liệu Cranfield ---
# Hàm này sẽ được Streamlit cache để chỉ chạy một lần
@st.cache_data
def load_cranfield_data():
    CRAN_DATA_DIR = "./Cranfield/" # Giả sử bạn đã giải nén Cranfield.zip vào cùng thư mục với script này, hoặc cấu hình đường dẫn tuyệt đối
    CRAN_QREL_FILE = os.path.join(CRAN_DATA_DIR, "cran_qrel.txt")
    CRAN_QRY_FILE = os.path.join(CRAN_DATA_DIR, "cran_qry.txt")
    CRAN_DOC_FILE = os.path.join(CRAN_DATA_DIR, "cran_all.txt")

    qrels_dict = defaultdict(list)
    try:
        with open(CRAN_QREL_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split()
                query_id = int(parts[0])
                doc_id = int(parts[1])
                qrels_dict[query_id].append(doc_id)
        # st.write(f"Loaded {len(qrels_dict)} queries with relevance judgments into qrels_dict.")
    except FileNotFoundError:
        st.error(f"Error: {CRAN_QREL_FILE} not found. Please ensure the path '{CRAN_DATA_DIR}' is correct and files exist.")
        st.stop() # Dừng ứng dụng nếu không tìm thấy file

    documents_raw = []
    try:
        with open(CRAN_DOC_FILE, 'r') as f:
            content = f.read()
            docs_raw = content.split('.I ')[1:]
            for d_raw in docs_raw:
                lines = d_raw.strip().split('\n')
                doc_text = ""
                reading_text = False
                for line in lines[1:]:
                    if line.strip() == '.W':
                        reading_text = True
                        continue
                    if reading_text:
                        if line.startswith('.'):
                            break
                        doc_text += line.strip() + " "
                documents_raw.append(doc_text.strip())
        # st.write(f"Loaded {len(documents_raw)} raw documents.")
    except FileNotFoundError:
        st.error(f"Error: {CRAN_DOC_FILE} not found. Please ensure the path '{CRAN_DATA_DIR}' is correct and files exist.")
        st.stop() # Dừng ứng dụng nếu không tìm thấy file
    
    N_DOCS = len(documents_raw)
    return documents_raw, N_DOCS, qrels_dict # (qrels_dict không dùng trong app nhưng trả về cho đầy đủ)

# --- 2. Định nghĩa các hàm tiền xử lý (như bạn đã có) ---
Stop_Words = stopwords.words("english")
porter_stemmer = PorterStemmer()

def tokenize(text):
    clean_txt = re.sub('[^a-z\s]+',' ',text)
    clean_txt = re.sub('(\s+)',' ',clean_txt)
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(clean_txt))
    words = [word for word in words if word not in Stop_Words]
    words = filter(lambda t: len(t)>=min_length, words)
    tokens = (list(map(lambda token: porter_stemmer.stem(token),words)))
    return tokens

# --- 3. Tải và Khởi tạo Mô hình BM25 (chỉ chạy một lần) ---
@st.cache_resource # st.cache_resource dùng cho các đối tượng phức tạp như model
def load_bm25_model(documents):
    st.write("Đang khởi tạo BM25 Model... (Chỉ chạy lần đầu)")
    corpus_tokenized = [tokenize(doc) for doc in documents]
    bm25_model = BM25Okapi(corpus_tokenized, k1=2.0, b=0.75) # Sử dụng tham số tốt nhất của bạn
    st.write("BM25 Model đã sẵn sàng.")
    return bm25_model

# --- 4. Định nghĩa hàm get_ranked_documents cho ứng dụng web (chỉ dùng BM25) ---
def get_ranked_documents_for_app(query_text, bm25_model_instance, N_DOCS_val):
    doc_sim = {}
    query_tokens = tokenize(query_text)

    if not query_tokens: # Xử lý truy vấn rỗng sau tokenize
        return []

    bm25_raw_scores = bm25_model_instance.get_scores(query_tokens)
    
    # Tạo dictionary điểm số
    for item in range(N_DOCS_val):
        doc_sim[item + 1] = bm25_raw_scores[item]

    # Lọc bỏ các tài liệu có điểm số 0 hoặc rất nhỏ và sắp xếp
    ranked_docs = sorted([(doc_id, score) for doc_id, score in doc_sim.items() if score > 1e-6],
                         key=operator.itemgetter(1), reverse=True)
    return ranked_docs

# ==============================================================================
# BẮT ĐẦU ỨNG DỤNG STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Cranfield Search Engine (BM25)", layout="centered")

st.title("🔎 Cranfield Search Engine")
st.markdown("Một công cụ tìm kiếm đơn giản cho bộ dữ liệu Cranfield.")

# Tải dữ liệu và mô hình một lần
documents_raw, N_DOCS, qrels_dict = load_cranfield_data()
bm25_model_instance = load_bm25_model(documents_raw)


# Ô nhập truy vấn
query = st.text_input("Nhập truy vấn của bạn vào đây:", placeholder="e.g., aerodynamic characteristics of wings")

# Nút tìm kiếm
if st.button("Tìm kiếm"):
    if query:
        st.subheader(f"Kết quả cho: \"{query}\"")
        
        # Thực hiện tìm kiếm
        results = get_ranked_documents_for_app(query, bm25_model_instance, N_DOCS)
        
        if results:
            st.write(f"Tìm thấy {len(results)} kết quả. Hiển thị 10 kết quả hàng đầu.")
            for i, (doc_id, score) in enumerate(results[:10]):
                if 0 < doc_id <= len(documents_raw):
                    doc_content = documents_raw[doc_id - 1]
                    snippet = doc_content[:400] + "..." if len(doc_content) > 400 else doc_content
                    
                    st.markdown(f"---")
                    st.write(f"**Hạng {i+1}: Tài liệu ID {doc_id} (Điểm số: {score:.4f})**")
                    st.markdown(f"```\n{snippet}\n```") # Hiển thị snippet trong khối code
                else:
                    st.warning(f"Hạng {i+1}: Tài liệu ID {doc_id} (Điểm số: {score:.4f}) - Không tìm thấy nội dung tài liệu gốc.")
        else:
            st.info("Không tìm thấy tài liệu nào khớp với truy vấn của bạn.")
    else:
        st.warning("Vui lòng nhập truy vấn để tìm kiếm.")

st.markdown("---")
st.write("Đồ án môn học Hệ thống truy xuất thông tin")