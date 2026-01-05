# 🤟 GPT API 기반 수어 동작 실시간 번역 서버

## 📌 프로젝트 개요
카메라 입력을 통해 수어 동작을 인식하고,
딥러닝 기반 수어 인식 결과를 GPT API와 연동하여 자연어 문장으로 실시간 번역하는 서버입니다.
- 수어 동작 인식 → 단어/토큰 추출
- 추출된 결과를 GPT 모델로 전달
- 사용자에게 자연스러운 문장 형태의 번역 결과 제공


## 🛠 기술 스택

**OpenAI Model** : <img src="https://img.shields.io/badge/gpt--3.5--turbo-412991?style=flat-square&logo=openai&logoColor=white"/> <img src="https://img.shields.io/badge/gpt--4o-412991?style=flat-square&logo=openai&logoColor=white"/>

**Language** : <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white"/> <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white"/> <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black"/>

**Framework & Library** : <img src="https://img.shields.io/badge/Django-092E20?style=flat-square&logo=django&logoColor=white"/> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/YOLOv5-000000?style=flat-square"/>


## ⚙️ 주요 기능
- 실시간 수어 동작 입력 처리
- YOLO 기반 손/동작 검출
- 딥러닝 모델을 통한 수어 단어 인식
- GPT API 연동을 통한 문장 단위 번역
- 웹 기반 번역 결과 출력


## 🔄 시스템 처리 흐름
1. 클라이언트에서 카메라 영상 입력
2. YOLO 기반 손 및 동작 검출
3. TensorFlow / PyTorch 모델로 수어 동작 분류
4. 분류된 단어 시퀀스를 GPT API에 전달
5. GPT가 자연어 문장으로 변환
6. 번역 결과를 클라이언트에 실시간 반환


## 🖥️ 실행 화면
    
### - 수어 번역
<img width="2876" height="1621" alt="Image" src="https://github.com/user-attachments/assets/2a94002d-4417-4c78-a94f-c21860f738d0" />


![Image](https://github.com/user-attachments/assets/5031cf6d-3b32-4b5d-8d62-bc2f9e0be7f4)
