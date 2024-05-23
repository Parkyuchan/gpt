from django.shortcuts import render

from django.shortcuts import render 
from django.http import JsonResponse 
import openai
import os

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content    

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

def query_view(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        prompt = str(prompt)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, 'data', 'example.txt')
        content = read_text_file(file_path) + "\n지금까지 나온 단어들을 중복된 단어가 있다면 한 단어만 사용해서 한글로 번역 후 자연스러운 문장으로 만들어줘"
        #response = get_completion(prompt)
        response = get_completion(content)
        return JsonResponse({'response': response})
    return render(request, 'index.html')