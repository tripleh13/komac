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
        f"{item.get('상세 정보', 'NONE')[:30] if item.get('상세 정보') else 'NONE'}"
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

def create_valid_tour_list(tour_data, gender, age, nature, type_pref):
    valid_tour_list = [
        item for item in tour_data 
        if item.get("tourId", "").startswith("TO")  # Check if tourId starts with TO
    ]

    if len(valid_tour_list) < 6:
        print("\n--- Adding similar tourIds for subject ---")
        additional_tours = find_similar_tours(
            tour_data=tour_data, 
            num_needed=6 - len(valid_tour_list), 
            gender=gender, 
            age=age, 
            nature=nature, 
            type_pref=type_pref
        )
        valid_tour_list.extend(additional_tours)
    
    return valid_tour_list[:9]  # Return up to 9 unique tourIds



# 유사한 tourId를 찾는 함수 (예시 함수, 실제 구현에 따라 변경 가능)
def find_similar_tours(tour_data, num_needed, gender, age, nature, type_pref):
    additional_tours = []
    search_keywords = f"{gender} {age} {nature} {type_pref}"
    
    # Search vector DB using keywords
    similar_docs = vector_db.similarity_search(search_keywords, k=30)
    
    print(f"\n--- 검색된 벡터 데이터 (검색어: {search_keywords}) ---")
    for doc in similar_docs:
        print(doc.page_content)

    # Adding similar tourIds based on keywords and avoiding duplicates
    for doc in similar_docs:
        if len(additional_tours) >= num_needed:
            break
        if "TO" in doc.page_content:
            tour_id = doc.page_content.split(".")[0]
            # Check for duplicates and log removal if necessary
            if {"tourId": tour_id} not in additional_tours:
                additional_tours.append({"tourId": tour_id})
            else:
                print(f"Duplicate tourId removed: {tour_id}")

    return additional_tours[:num_needed]

import random

# 주제 제목 목록
subject_titles = [
    "고흥의 숨겨진 보석을 찾아 떠나는 여정", 
    "고흥에서 힐링 여행", 
    "바다와 산이 어우러진 고흥의 매력 탐방", 
    "고흥의 멋진 풍경과 함께하는 하루", 
    "문화와 전통이 살아 숨 쉬는 고흥 여행", 
    "고흥의 대표 명소를 스탬프투어로 즐겨보세요"
]

# LLM에서 반환된 JSON 검증 및 필터링
def filter_tour_ids(parsed_json, vector_db):
    filtered_stampList = []
    
    # 벡터 데이터에서 모든 tourId를 추출하여 집합으로 저장
    all_vector_tourIds = set(
        doc.page_content.split(".")[0]
        for doc in vector_db.similarity_search("TO", k=1000)  # 모든 tourId 조회
        if "TO" in doc.page_content
    )
    
    for subject_data in parsed_json.get("stampList", []):
        # 영어 또는 한자가 포함되었는지 확인
        if any(ord(char) > 128 for char in subject_data["subject"]):
            # 중복 없이 랜덤한 주제 선택
            subject_data["subject"] = random.choice(subject_titles)
        
        valid_tourList = []
        for tour in subject_data.get("tourList", []):
            if tour["tourId"] in all_vector_tourIds:
                valid_tourList.append(tour)
        
        # 유효한 tourList가 있을 때만 추가
        if valid_tourList:
            subject_data["tourList"] = valid_tourList
            filtered_stampList.append(subject_data)
    
    parsed_json["stampList"] = filtered_stampList
    return parsed_json

@app.route('/ai_stamp', methods=['GET'])
async def ask_question():
    global vector_db  # 이미 생성된 벡터 DB 사용

    # 요청 수신 시 타이머 시작
    start_time = time.time()

    # 쿼리 매개변수 가져오기
    gender = request.args.get('gender', 'NONE')
    age = request.args.get('age', 'NONE')
    nature = request.args.get('nature', 'NONE')
    type_pref = request.args.get('type', 'NONE')

    # LLM prompt 설정
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

    print("\n=== LLM으로 보낼 question ===")
    print(message)

    # 벡터 데이터 검색
    docs = vector_db.similarity_search(message, k=30)
    print("\n=== 검색된 벡터 데이터 ===")
    for doc in docs:
        print(doc.page_content)

    # LLM prompt 생성
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

    elapsed_time_search = print_elapsed_time(start_time, "vector data search")

    # LLM 응답 생성
    answer = llm.invoke(full_prompt)
    print("\n=== LLM에서 반환된 answer ===")
    print(answer)

    # JSON 응답에서 serverElapsedTime 추가
    try:
        json_string = extract_json(answer)
        if not json_string:
            return jsonify({'code': '99', 'message': 'No valid JSON found.', 'serverElapsedTime': elapsed_time_search})

        json_string = fix_json_string(json_string)
        if not json_string:
            return jsonify({'code': '99', 'message': 'Invalid JSON structure after fixing.', 'serverElapsedTime': elapsed_time_search})

        parsed_json = json.loads(json_string)

        # 유효 tourId 및 tourList 정리
        existing_tour_ids = set()
        for stamp in parsed_json.get('stampList', []):
            unique_tour_list = []
            for tour in stamp.get('tourList', []):
                tour_id = tour["tourId"]
                if tour_id not in existing_tour_ids:
                    unique_tour_list.append(tour)
                    existing_tour_ids.add(tour_id)

            # 부족한 tourId 추가
            if len(unique_tour_list) < 6:
                similar_tours = find_similar_tours(stamp, 6 - len(unique_tour_list), gender, age, nature, type_pref, existing_tour_ids)
                unique_tour_list.extend(similar_tours)

            if len(unique_tour_list) < 6:
                random_tours = find_random_tours(6 - len(unique_tour_list), existing_tour_ids)
                unique_tour_list.extend(random_tours)

            for idx, tour in enumerate(unique_tour_list[:9]):  # 최대 9개
                tour['orders'] = idx

            stamp['tourList'] = unique_tour_list[:9]

        if 'code' not in parsed_json or 'stampList' not in parsed_json:
            return jsonify({'code': '98', 'message': 'Invalid JSON structure: Missing required keys.', 'serverElapsedTime': elapsed_time_search})

        elapsed_time_response = print_elapsed_time(start_time, "JSON result output")

        parsed_json["serverElapsedTime"] = elapsed_time_response  # 서버 측 처리 시간을 응답에 추가

        return jsonify(parsed_json)

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        return jsonify({
            'code': '99',
            'serverElapsedTime': elapsed_time_search,
            'message': f'Response is not valid JSON. Error: {str(e)}',
            'error_detail': {
                'position': e.pos,
                'message': e.msg
            }
        })



def find_similar_tours(stamp, num_needed, gender, age, nature, type_pref, existing_tour_ids):
    additional_tours = []
    search_keywords = f"{gender} {age} {nature} {type_pref}"
    matched_tours = []  # 우선순위별로 정렬하기 위한 임시 리스트

    # 벡터 데이터 검색
    similar_docs = vector_db.similarity_search(search_keywords, k=30)
    print(f"\n--- 검색된 벡터 데이터 (검색어: {search_keywords}) ---")
    
    for doc in similar_docs:
        print(doc.page_content)

    # 검색된 tourId 중 일치 키워드 수에 따라 추가할 리스트 생성
    for doc in similar_docs:
        if "TO" in doc.page_content:
            tour_id = doc.page_content.split(".")[0]

            # 중복 확인 후 각 키워드의 값 확인
            if tour_id not in existing_tour_ids:
                matched_keywords = []
                match_count = 0  # 일치 키워드 수

                if gender in doc.page_content:
                    matched_keywords.append(f"gender={gender}")
                    match_count += 1
                if age in doc.page_content:
                    matched_keywords.append(f"age={age}")
                    match_count += 1
                if nature in doc.page_content:
                    matched_keywords.append(f"nature={nature}")
                    match_count += 1
                if type_pref in doc.page_content:
                    matched_keywords.append(f"type_pref={type_pref}")
                    match_count += 1
                
                # 매칭된 키워드와 함께 tourId와 매칭 수 저장
                matched_tours.append((tour_id, matched_keywords, match_count))

    # 매칭된 키워드 수가 많은 순서대로 정렬
    matched_tours = sorted(matched_tours, key=lambda x: x[2], reverse=True)

    # 필요 수량까지 추가하고, 일치한 키워드 출력
    for tour_id, matched_keywords, match_count in matched_tours:
        if len(additional_tours) >= num_needed:
            break
        print(f"추가된 tourId: {tour_id}, 일치한 키워드: {', '.join(matched_keywords)}")
        additional_tours.append({"tourId": tour_id})
        existing_tour_ids.add(tour_id)  # 중복 방지용 set에 추가

    return additional_tours[:num_needed]


def find_random_tours(num_needed, existing_tour_ids):
    random_tours = []
    random_docs = vector_db.similarity_search("random", k=50)
    for doc in random_docs:
        if len(random_tours) >= num_needed:
            break
        if "TO" in doc.page_content:
            tour_id = doc.page_content.split(".")[0]
            if {"tourId": tour_id} not in existing_tour_ids:
                random_tours.append({"tourId": tour_id})
                existing_tour_ids.add(tour_id)  # Track this ID as added
                print(f"Added random tourId: {tour_id}")
            else:
                print(f"Duplicate random tourId skipped: {tour_id}")
                
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
    return elapsed  # 경과 시간을 반환하도록 수정


if __name__ == '__main__':
    # 프로그램 시작 시간 기록
    program_start_time = time.time()
    print(f"Program started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize the event loop and vector database before starting the server
    loop = asyncio.get_event_loop()
    
    # 벡터 DB 초기화 시간 기록
    vector_db_start_time = time.time()
    vector_db = loop.run_until_complete(initialize_vector_database())
    vector_db_end_time = time.time()
    
    print(f"Vector DB initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Vector DB loading time: {vector_db_end_time - vector_db_start_time:.2f} seconds")
    
    # Flask 서버 실행 시간 기록
    server_start_time = time.time()
    app.run(host='0.0.0.0', port=5000)
    print(f"Flask server ready at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time from program start to server ready: {server_start_time - program_start_time + (time.time() - server_start_time):.2f} seconds")
