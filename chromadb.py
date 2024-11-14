import requests
import aiohttp  # 비동기 HTTP 요청을 처리하기 위한 라이브러리
import asyncio  # 비동기 작업을 처리하기 위한 라이브러리
from flask import Flask, request, jsonify
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import re
import os
import PyPDF2

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
                {"tourId": "관광지 ID", "orders": 1},
                {"tourId": "관광지 ID", "orders": 2},
                {"tourId": "관광지 ID", "orders": 3},
                {"tourId": "관광지 ID", "orders": 4},
                {"tourId": "관광지 ID", "orders": 5},
                {"tourId": "관광지 ID", "orders": 6}
            ]
        }
    ]
}

# PDF 파일 경로 설정
pdf_directory = "Data/pdf/"

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

# 벡터 데이터베이스 생성 함수 (기존)
def create_vector_database(api_data, pdf_texts):
    # API 데이터 처리
    json_texts = [
        f"{item.get('관광지 ID', 'NONE')}. "
        f"{item.get('첫번째 구분', 'NONE')}. "
        f"{item.get('두번째 구분', 'NONE')}. "
        f"{item.get('권역 구분', 'NONE')}. "
        f"{item.get('고흥 10경 여부', 'NONE')}. "
        f"{item.get('AI 참조데이터', 'NONE')}. "
        f"{item.get('명칭', 'NONE')}. "
        f"{item.get('상세 정보', 'NONE')}"
        for item in api_data['data']
    ] if api_data else []

    # PDF 데이터와 API 데이터 결합
    combined_texts = json_texts + pdf_texts

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    split_texts = text_splitter.split_text(" ".join(combined_texts))

    # 벡터 데이터베이스 생성 /all-MiniLM-L6-2v, all-mpnet-base-v2
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vector_store = Chroma.from_texts(split_texts, embedding=embeddings)

    return vector_store

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

# 서버 시작 전에 벡터 DB를 미리 초기화
async def initialize_vector_database():
    pdf_texts = await extract_text_from_pdfs(pdf_directory)
    json_data = await fetch_json_data(api_url)
    return create_vector_database(json_data, pdf_texts)

# 이미 생성된 벡터 DB를 저장할 전역 변수
vector_db = None

@app.route('/ai_stamp', methods=['GET'])
async def ask_question():
    global vector_db  # 이미 생성된 벡터 DB 사용

    gender = request.args.get('gender', 'NONE')
    age = request.args.get('age', 'NONE')
    nature = request.args.get('nature', 'NONE')
    type_pref = request.args.get('type', 'NONE')

    message = "Search for the tourId corresponding to the following recommended words. The tourId is data from the vector database that starts with 'TO'. The search for the tourId should consider each block of content between a starting 'TO' and the next 'TO', and its similarity to the following words: '" + gender + "', '" + age + "', '" + nature + "', and '" + type_pref + "'. The tourId should be retrieved and output only if there is information in the vector database starting with the 'TO' code. If no such information exists, it should not be output. The output must be in a key-based JSON structure. The data related to the subject should be presented as a complete and well-structured sentence appropriate to the topic."
    system_role01 = "Generate a JSON data that returns the value of code as 00. The stampList key has a structure where it holds values as an array of subject and tourList keys. The tourId represents the tourist ID, and orders refers to the index number inside the tourList array."
    system_role02 = "Please generate the stampList data with 3 subjects and tourList arrays. Each tourList should contain between 6 and 9 data entries, with optimal data generated for each list. Specifically, there should be 3 pairs of tourList, with each list containing between 6 and 9 tourId and orders entries."
    system_role03 = "Please generate the tourList arrays with a maximum of 9 data entries per array, ensuring that the orders sequence starts at 0 within each array. Additionally, the JSON format must be correctly structured to ensure there are no errors and that it is a valid JSON output."
    example = system_role01 + system_role02 + system_role03

    output_example_str = json.dumps(output_exam01, ensure_ascii=False)
    question = f"{message}. {example}. {output_example_str}"
    print("\n=== LLM으로 보낼 question ===")
    print(question)

    # 기존 벡터 DB를 활용하여 유사 질문 처리
    docs = vector_db.similarity_search(question, k=10)
    context = "\n\n".join([doc.page_content for doc in docs])
    full_prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    answer = llm.invoke(full_prompt)

    print("\n=== LLM에서 반환된 answer ===")
    print(answer)

    try:
        json_string = extract_json(answer)
        if not json_string:
            return jsonify({'code': '99', 'message': 'No valid JSON found.'})
        json_string = fix_json_string(json_string)
        if not json_string:
            return jsonify({'code': '99', 'message': 'Invalid JSON structure after fixing.'})

        parsed_json = json.loads(json_string)
        for stamp in parsed_json.get('stampList', []):
            for idx, tour in enumerate(stamp.get('tourList', [])):
                tour['orders'] = idx  # Reset orders to start at 0

        if 'code' not in parsed_json or 'stampList' not in parsed_json:
            return jsonify({'code': '98', 'message': 'Invalid JSON structure: Missing required keys.'})

        return jsonify(parsed_json)
    
    except json.JSONDecodeError:
        return jsonify({'code': '99', 'message': 'Response is not valid JSON.'})

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    vector_db = loop.run_until_complete(initialize_vector_database())  # 서버 시작 전에 DB 초기화
    app.run(host='0.0.0.0', port=5000)
