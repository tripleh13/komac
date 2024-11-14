import requests
import aiohttp  # 비동기 HTTP 요청을 처리하기 위한 라이브러리
import asyncio  # 비동기 작업을 처리하기 위한 라이브러리
from flask import Flask, request, jsonify
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings  # 업데이트된 HuggingFaceEmbeddings import
from langchain.vectorstores import FAISS  # FAISS를 사용한 벡터 저장소
import torch  # PyTorch for GPU support
import json
import re
import os
import PyPDF2
import time  # For measuring time

app = Flask(__name__)

# Ollama LLM 설정 (Gemma 2B 모델) - system role 추가
llm = Ollama(
    model="gemma2:2b",
    system="This is a useful assistant who can provide concise and accurate answers to questions based on searched documents."
)

# API URL 설정
api_url = "https://goheungtour.witches.co.kr/api/getAllTourDataUsedStamp.do"

# Example output JSON 데이터
output_exam01 = {
    "code": "00",
    "stampList": [
        {
            "subject": "고흥군 추천 스탬프 투어",
            "tourList": [
                {"tourId": "관광지 ID", "orders": 0},
                {"tourId": "관광지 ID", "orders": 1},
                {"tourId": "관광지 ID", "orders": 2},
                {"tourId": "관광지 ID", "orders": 3},
                {"tourId": "관광지 ID", "orders": 4},
                {"tourId": "관광지 ID", "orders": 5},
                {"tourId": "관광지 ID", "orders": 6},
                {"tourId": "관광지 ID", "orders": 7},
                {"tourId": "관광지 ID", "orders": 8}
            ]
        }
    ]
}

# PDF 파일 경로 설정
pdf_directory = "Data/pdf/"

# Use GPU (CUDA) for HuggingFace Embeddings if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# PDF 파일에서 텍스트를 추출하는 함수 (비동기 처리)
async def extract_text_from_pdfs(pdf_directory):
    pdf_texts = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
                pdf_texts.append(text)
    return pdf_texts

# 비동기 API 호출을 위한 함수
async def fetch_json_data(api_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    json_data = await response.json()
                    print("=== API에서 반환된 JSON 데이터 구조 ===")
                    print(json_data)
                    return json_data
                else:
                    print(f"API 호출 오류: {response.status}")
                    return None
    except aiohttp.ClientError as e:
        print(f"API 호출 오류: {e}")
        return None

from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
import torch

def create_vector_database(api_data, pdf_texts):
    # API 데이터 처리
    json_texts = [
        f"{item.get('관광지 ID', 'NONE')}. "
        f"{item.get('첫번째 구분', 'NONE')}. "
        f"{item.get('두번째 구분', 'NONE')}. "
        f"{item.get('AI 참조데이터', 'NONE')}. "
        f"{item.get('명칭', 'NONE')}. "
        f"{item.get('상세 정보', 'NONE')[:80] if item.get('상세 정보') else 'NONE'}"
        for item in api_data['data']
    ] if api_data else []

    # PDF 데이터와 API 데이터 결합
    combined_texts = " ".join(json_texts + pdf_texts)

    # TO로 시작하는 단위로 텍스트를 분할
    split_texts = re.split(r'(TO\d+)', combined_texts)
    grouped_texts = ["".join(split_texts[i:i+2]) for i in range(1, len(split_texts), 2)]

    # 임베딩 생성
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # FAISS 데이터베이스 생성 (텍스트 단위로 벡터화)
    vector_store = FAISS.from_texts(grouped_texts, embeddings)

    return vector_store


# JSON 추출 및 검증
def extract_json(text):
    json_pattern = re.compile(r'{.*}', re.DOTALL)
    json_match = json_pattern.search(text)
    if json_match:
        return json_match.group(0)
    return None

# 서버 시작 전에 벡터 DB를 미리 초기화
async def initialize_vector_database():
    pdf_texts = await extract_text_from_pdfs(pdf_directory)
    json_data = await fetch_json_data(api_url)
    return create_vector_database(json_data, pdf_texts)

# 이미 생성된 벡터 DB를 저장할 전역 변수
vector_db = None

# 유효한 tourId 리스트 생성 함수
def create_valid_tour_list(tour_data):
    valid_tour_list = [
        item for item in tour_data 
        if item.get("tourId", "").startswith("TO")  # tourId가 TO로 시작하는지 확인
    ]
    
    if len(valid_tour_list) < 6:
        # 유사한 tourId를 추가해서 최소 6개가 되도록 보장
        additional_tours = find_similar_tours(tour_data, num_needed=6 - len(valid_tour_list))
        valid_tour_list.extend(additional_tours)
    
    return valid_tour_list[:9]  # 6개 이상, 최대 9개까지만 반환

# 유사한 tourId를 찾는 함수 (예시 함수, 실제 구현에 따라 변경 가능)
def find_similar_tours(tour_data, num_needed):
    # 더 많은 tourId를 찾는 로직, 일단 num_needed 수만큼 tourId를 반환
    additional_tours = [
        item for item in tour_data 
        if not item.get("tourId", "").startswith("TO")
    ]
    return additional_tours[:num_needed]

@app.route('/ai_stamp', methods=['GET'])
async def ask_question():
    global vector_db  # Already created vector DB is used.

    # Request received, start timing
    start_time = time.time()

    gender = request.args.get('gender', 'NONE')
    age = request.args.get('age', 'NONE')
    nature = request.args.get('nature', 'NONE')
    type_pref = request.args.get('type', 'NONE')

    # LLM prompt with all conditions clearly outlined.
    message = """
    Find the 'tourId' that matches the following keywords. The 'tourId' **must** exist in the vector database and must start with 'TO'. Do **not** create or generate new 'tourId'. Use **only** tourIds that exist in the database.

    The output must be a valid JSON structure and follow the exact format described below.

    **Requirements**:
    1. The 'stampList' must contain exactly **3 unique 'subjects'**, each representing a distinct **theme** based on the common attributes or detailed descriptions of the tourIds. The subject titles must be concise and specific tour course titles written in **Korean**.
    
    2. For each 'subject', the 'tourList' must contain **between 6 and 9 unique tourIds**. Ensure that each 'tourList' contains **at least 6 tourIds**. If fewer than 6 unique tourIds are available, search the vector database for the most **similar** tourIds and **ensure at least 6 unique tourIds** are included.
    
    3. **No 'tourId'** should be duplicated across different subjects or within the same 'tourList'.

    4. If any of the above conditions are not met, return a structured error message indicating which condition was violated.

    5. The JSON format must be valid, following proper syntax including the use of commas, brackets, and braces.
    """

    # Fetch data from vector DB (increase k to 50 for more search results)
    print("\n=== LLM으로 보낼 question ===")
    print(message)

    docs = vector_db.similarity_search(message, k=50)
    print("\n=== 검색된 벡터 데이터 ===")
    for doc in docs:
        print(doc.page_content)

  # LLM prompt creation
    context = ". ".join([doc.page_content for doc in docs])
    full_prompt = f"""
    Context: {context}

    Based on the tourId details provided in the context, generate 3 concise and meaningful Korean subject titles that represent unique themes for a tour course. 
    Consider the provided keywords ({gender}, {age}, {nature}, {type_pref}) along with the tourId descriptions to ensure that each title captures the essence of the locations or experiences. 
    The output must follow this exact JSON format:

    {{
        "code": "00",
        "stampList": [
            {{
                "subject": "Korean title for theme 1",
                "tourList": [
                    {{"tourId": "TOxxxxxx", "orders": 0}},
                    ...
                ]
            }},
            {{
                "subject": "Korean title for theme 2",
                "tourList": [
                    {{"tourId": "TOxxxxxx", "orders": 0}},
                    ...
                ]
            }},
            {{
                "subject": "Korean title for theme 3",
                "tourList": [
                    {{"tourId": "TOxxxxxx", "orders": 0}},
                    ...
                ]
            }}
        ]
    }}

    Ensure that the output is a valid JSON structure. If there are any issues, return a structured error message indicating the problem.
    """

    print("\n=== LLM으로 보낼 full_prompt ===")
    print(full_prompt)

    print_elapsed_time(start_time, "vector data search")

    # LLM answer generation
    answer = llm.invoke(full_prompt)
    print("\n=== LLM에서 반환된 answer ===")
    print(answer)

    # Extract JSON from the LLM response
    try:
        json_string = extract_json(answer)
        if not json_string:
            return jsonify({'code': '99', 'message': 'No valid JSON found.'})
        
        json_string = fix_json_string(json_string)
        if not json_string:
            return jsonify({'code': '99', 'message': 'Invalid JSON structure after fixing.'})

        parsed_json = json.loads(json_string)

        # Check if there are fewer than 3 subjects and add default subject if necessary
        if len(parsed_json.get('stampList', [])) < 3:
            missing_subjects = 3 - len(parsed_json['stampList'])
            for _ in range(missing_subjects):
                parsed_json['stampList'].append({
                    "subject": "고흥군 추천 관광, 여행, 힐링, 맛집 투어",
                    "tourList": []  # Will be populated below
                })

        # Fill missing tourIds and print similar/random tourIds
        for stamp in parsed_json.get('stampList', []):
            if len(stamp.get('tourList', [])) < 6:
                print(f"\n--- Adding similar tourIds for subject: {stamp['subject']} ---")
                similar_tours = find_similar_tours(stamp, 6 - len(stamp['tourList']), gender, age, nature, type_pref)
                print(f"Similar tourIds: {similar_tours}")
                stamp['tourList'].extend(similar_tours)

            # If still less than 6, fill with random tours
            if len(stamp.get('tourList', [])) < 6:
                print(f"\n--- Adding random tourIds for subject: {stamp['subject']} ---")
                random_tours = find_random_tours(6 - len(stamp['tourList']))
                print(f"Random tourIds: {random_tours}")
                stamp['tourList'].extend(random_tours)

            # Assign 'orders' index to each tourId
            for idx, tour in enumerate(stamp.get('tourList', [])):
                tour['orders'] = idx  # Reset orders to start at 0

        # Ensure JSON structure correctness
        if 'code' not in parsed_json or 'stampList' not in parsed_json:
            return jsonify({'code': '98', 'message': 'Invalid JSON structure: Missing required keys.'})

        
        # Log the time after JSON result output
        print_elapsed_time(start_time, "JSON result output")

        return jsonify(parsed_json)

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        return jsonify({
            'code': '99',
            'message': f'Response is not valid JSON. Error: {str(e)}',
            'error_detail': {
                'position': e.pos,
                'message': e.msg
            }
        })


# Helper function to find additional similar tours based on the request inputs
def find_similar_tours(stamp, num_needed, gender, age, nature, type_pref):
    additional_tours = []
    # Filtering logic based on gender, age, nature, or type
    # Example filtering logic: check if 'gender', 'age', etc. match in vector DB content
    for doc in vector_db.similarity_search(f"{gender} {age} {nature} {type_pref}", k=50):
        if len(additional_tours) >= num_needed:
            break
        # Extract valid tourId from the document content
        if "TO" in doc.page_content:
            additional_tours.append({"tourId": doc.page_content.split(".")[0]})
    
    return additional_tours[:num_needed]


# Helper function to find random tourIds when needed
def find_random_tours(num_needed):
    random_tours = []
    random_docs = vector_db.similarity_search("random", k=50)
    for doc in random_docs:
        if len(random_tours) >= num_needed:
            break
        if "TO" in doc.page_content:
            random_tours.append({"tourId": doc.page_content.split(".")[0]})
    
    return random_tours[:num_needed]

# JSON 추출 및 검증
def extract_json(text):
    json_pattern = re.compile(r'{.*}', re.DOTALL)
    json_match = json_pattern.search(text)
    if json_match:
        return json_match.group(0)
    return None

def fix_json_string(json_string):
    json_string = json_string.strip()
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
    try:
        json.loads(json_string)
        return json_string
    except json.JSONDecodeError as e:
        print(f"JSON decode error at {e.pos}: {e.msg}")
        return None

# 타이밍 측정을 위한 유틸리티 함수
def print_elapsed_time(start_time, stage):
    elapsed = time.time() - start_time
    print(f"Elapsed time after {stage}: {elapsed:.2f} seconds")


if __name__ == '__main__':
    # Initialize the event loop and vector database before starting the server
    loop = asyncio.get_event_loop()
    vector_db = loop.run_until_complete(initialize_vector_database())  # Pre-initialize the vector DB before server starts

    # Run the Flask app to handle incoming requests
    app.run(host='0.0.0.0', port=5000)
