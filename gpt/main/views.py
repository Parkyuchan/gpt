from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import openai
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os
import atexit

# 모델 로드
right_model = tf.keras.models.load_model('main/models/best_right_model.keras')
left_model = tf.keras.models.load_model('main/models/best_left_model.keras')

# 동작 정의
actions_right = ['meet', 'nice', 'hello', 'you', 'name', 'what', 'have', 'do not have', 'me']
actions_left = ['meet', 'nice', 'hello', 'you', 'name', 'what', 'have', 'do not have', 'me']
actions_both = ['meet', 'nice', 'hello', 'you', 'name', 'what', 'have', 'do not have', 'me']

# 시퀀스 길이 설정
seq_length = 30
origin_cap = 0

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
def hand_gesture(request):
    user_id = request.session.session_key
    file_path = get_unique_user_file_path(user_id)
    result_file_path = get_result_file_path(user_id)

    @atexit.register
    def cleanup():
        if os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("")
        if os.path.exists(result_file_path):
            with open(result_file_path, 'w', encoding='utf-8') as f:
                f.write("")

    def gen_frames():
        sequence = {'left': [], 'right': []}
        action_seq = []
        last_action_time = time.time()
        this_action = ''
        last_saved_action = ''

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)

        user_id = request.session.session_key
        if not user_id:
            request.session.create()
            user_id = request.session.session_key

        channel_layer = get_channel_layer()

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
         분'

def handle_prompt(request):
    if request.method == 'POST':
        user_id = request.session.session_key
        file_path = get_unique_user_file_path(user_id)
        result_file_path = get_result_file_path(user_id)
        handle_data = read_text_file(file_path)

        if handle_data:
            result = handle_data + "\n지금 위에 있는 영어 단어들 중 중복된 단어는 딱 한번씩만 사용해서 자연스러운 한문장이 되도록 만들고 한글로 해석해서 출력해줘"
            result_content = get_completion(result)
            append_to_file(result_file_path, result_content)

            # 파일을 초기화
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("")

            return JsonResponse({'prompt' : handle_data, 'response': result_content})
        else:
            return JsonResponse({'response': 'No gesture detected'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def index(request) :
    user_id = request.session.session_key
    if not user_id:
        request.session.create()
        user_id = request.session.session_key
    file_path_result = get_result_file_path(user_id)
    result_content = read_text_file(file_path_result)
    
    return render(request, 'index.html', {'result_content': result_content})


def get_unique_user_file_path(user_id):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_dir = os.path.join(base_dir, 'data', user_id)
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, 'example.txt')
    return file_path


# 24.05.31 gpt로 해석된 문장이 저장되는 경로
def get_result_file_path(user_id):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(base_dir, 'result', user_id)
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, 'getResponse.txt')
    return file_path

def append_to_file(file_path, text):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"{text}\n")
       
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return None

def get_completion(prompt):
    try:
        print(prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = response.choices[0].message['content']
        print(message)
        return message
    except openai.error.OpenAIError as e:
        if e.code == 'quota_exceeded':
            return "You have exceeded your quota. Please check your OpenAI plan and billing details."
        else:
            print(f"Error occurred: {e}")
            return "There was an error processing your request."
