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
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt



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
                break

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)
            current_time = time.time()

            if result.multi_hand_landmarks is not None:

                for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(hand_landmarks.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                    v = v2 - v1
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    dot_product = np.einsum('nt,nt->n', v, v)
                    dot_product = np.clip(dot_product, -1.0, 1.0)

                    angle = np.arccos(dot_product)
                    angle = np.degrees(angle)

                    d = np.concatenate([joint.flatten(), angle])

                    hand_label = hand_info.classification[0].label
                    if hand_label == 'Right':
                        sequence['right'].append(d)
                        if len(sequence['right']) > seq_length:
                            sequence['right'].pop(0)
                    else:
                        sequence['left'].append(d)
                        if len(sequence['left']) > seq_length:
                            sequence['left'].pop(0)

                if len(sequence['right']) == seq_length and len(sequence['left']) == seq_length:
                    input_data_right = np.expand_dims(np.array(sequence['right']), axis=0)
                    input_data_left = np.expand_dims(np.array(sequence['left']), axis=0)

                    y_pred_right = right_model.predict(input_data_right).squeeze()
                    y_pred_left = left_model.predict(input_data_left).squeeze()

                    i_pred_right = int(np.argmax(y_pred_right))
                    i_pred_left = int(np.argmax(y_pred_left))

                    conf_right = y_pred_right[i_pred_right]
                    conf_left = y_pred_left[i_pred_left]
                    
                    if i_pred_right < len(actions_right):
                        print(f"Right hand prediction: {actions_right[i_pred_right]} ({conf_right:.2f})")
                    else:
                        print(f"Right hand prediction index {i_pred_right} is out of range for actions_right")

                    if i_pred_left < len(actions_left):
                        print(f"Left hand prediction: {actions_left[i_pred_left]} ({conf_left:.2f})")
                    else:
                        print(f"Left hand prediction index {i_pred_left} is out of range for actions_left")

                    if conf_right > 0.5 and conf_left > 0.5:
                        if i_pred_right < len(actions_both) and i_pred_left < len(actions_both):
                            action = actions_both[i_pred_right]
                            action_seq.append(action)

                            if len(action_seq) > 3:
                                action_seq = action_seq[-3:]

                            if action_seq.count(action) > 1:
                                this_action = action
                            else:
                                this_action = ' '

                            last_action_time = current_time
                            sequence = {'left': [], 'right': []}
                
                elif len(sequence['right']) == seq_length:
                    input_data = np.expand_dims(np.array(sequence['right']), axis=0)
                    y_pred = right_model.predict(input_data).squeeze()
                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]
                    
                    if i_pred < len(actions_right):
                        print(f"Right hand prediction: {actions_right[i_pred]} ({conf:.2f})")

                    if conf > 0.5:
                        action = actions_right[i_pred]
                        action_seq.append(action)

                        if len(action_seq) > 3:
                            action_seq = action_seq[-3:]

                        if action_seq.count(action) > 1:
                            this_action = action
                        else:
                            this_action = ' '

                        last_action_time = current_time
                        sequence = {'left': [], 'right': []}
                        
                    else:
                        print(f"Right hand prediction index {i_pred} is out of range for actions_right")

                elif len(sequence['left']) == seq_length:
                    input_data = np.expand_dims(np.array(sequence['left']), axis=0)
                    y_pred = left_model.predict(input_data).squeeze()
                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]
                    
                    if i_pred < len(actions_left):
                        print(f"Left hand prediction: {actions_left[i_pred]} ({conf:.2f})")

                    if conf > 0.5:
                        action = actions_left[i_pred]
                        action_seq.append(action)

                        if len(action_seq) > 3:
                            action_seq = action_seq[-3:]

                        if action_seq.count(action) > 1:
                            this_action = action
                        else:
                            this_action = ' '

                        last_action_time = current_time
                        sequence = {'left': [], 'right': []}
                        
                    else:
                        print(f"Left hand prediction index {i_pred} is out of range for actions_left")

                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if current_time - last_action_time < 1:
                cv2.putText(img, this_action, org=(0, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                if this_action != last_saved_action:
                    last_action_time = time.time()
                
                    append_to_file(file_path, this_action)
                    last_saved_action = this_action

                # Send action to WebSocket
                async_to_sync(channel_layer.group_send)(
                    f"user_{user_id}",
                    {
                        "type": "action.message",
                        "action": this_action,
                    }
                )
            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        cap.release()

    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

openai.api_key='#'

def handle_prompt(request):
    if request.method == 'POST':
        user_id = request.session.session_key
        file_path = get_unique_user_file_path(user_id)
        result_file_path = get_result_file_path(user_id)
        handle_data = read_text_file(file_path)

        if handle_data:
            result = handle_data + "수어 문장으로 해석"
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
    start = time.time() # 시작
    user_id = request.session.session_key
    if not user_id:
        request.session.create()
        user_id = request.session.session_key
    file_path_result = get_result_file_path(user_id)
    result_content = read_text_file(file_path_result)
    
    
    return render(request, 'index.html', {'result_content': result_content})

@csrf_exempt
def ready(request):
    
    start = time.time()
    if request.method == 'POST':
        # 실행할 코드
        first_setting_path = get_setting_file_path()
        first_setting_content = read_text_file(first_setting_path)
        get_completion(first_setting_content, "gpt-4-turbo")
        
        third_setting_path = get_setting_file_path('third', 'third_setting.txt')
        third_setting_content = read_text_file(third_setting_path)
        get_completion(third_setting_content)
        
        time.sleep(1)
        print(f"{time.time() - start:.4f} sec")
        return JsonResponse({"status": "success"})
        
    return render(request, 'ready.html')
    

def get_setting_file_path(folder_name = 'first', file_name = 'first_setting.txt'):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, folder_name, file_name)
    return file_path

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

def get_completion(prompt, model = "gpt-3.5-turbo"):
    try:
        print(prompt)
        response = openai.ChatCompletion.create(
            model=model,
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
