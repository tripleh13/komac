from flask import Flask, request, jsonify
from datetime import datetime
from langchain_community.llms import Ollama

app = Flask(__name__)

# Ollama LLM 설정 (Gemma 2B 모델)
llm = Ollama(model="gemma:2b")

@app.route('/ask', methods=['GET'])
def ask_question():
    question = request.args.get('message')

    if question:
        # Ollama 모델로 질문을 보내서 답변을 받음
        answer = llm.invoke(question)
        
        # 현재 날짜와 시간 추가
        response_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({
            'question': question,
            'answer': answer,
            'response_time': response_time
        })
    else:
        return jsonify({'error': 'message 파라미터가 없습니다.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
