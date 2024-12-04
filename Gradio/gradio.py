import gradio as gr
import openai
from  openai import OpenAI
import base64
import os
import requests

# OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "https://dfqnnshbcpffgkps.tunnel-pt.elice.io/proxy/8007/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

fast_api_base = "http://localhost:8000/"

prompt_rag = "Please provide the major issues of 2024. It would be great if you could combine the information I provide with your own knowledge."

flag = 0

def main():

    models = client.models.list()
    model = models.data[0].id
    print(f"Using model: {model}")

    def encode_base64_content_from_file(image_path: str) -> str:
        """Encode a content retrieved from image_path to base64 format."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
    # Image URL and text input inference
    def run_image_and_text_inference(image_base64, question) -> str:
        # Constructing the messages with both text and image content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_rag + question},
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        ]

        # Send the combined input to the model
        chat_completion = client.chat.completions.create(
            model=model,  # Adjust model as per availability
            messages=messages,
            max_tokens=512,  # Configurable for more detailed responses
        )

        result = chat_completion.choices[0].message.content
        return result
    
    def run_text_inference(question) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt_rag + question
            }
        ]

        chat_completion = client.chat.completions.create(
            model=model, 
            messages=messages,
            max_tokens=512,
        )

        result = chat_completion.choices[0].message.content
        return result
    
    # 공통 인퍼런스 함수
    def run_inference(question, chat_history=None, image_path=None):
        messages = []

        # 이전 채팅 기록이 있는 경우 메시지 구성
        if chat_history:
            for msg in chat_history:
                if flag==1:
                    print('조건걸림!!!!')
                    # 이미지 경로를 포함한 경우, 인코딩된 이미지로 변환하여 messages에 추가
                    image_base64 = encode_base64_content_from_file(image_path)
                    messages.append({
                        "role": msg["role"],
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ],
                    })
                else:
                    # 텍스트 메시지인 경우 그대로 추가
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        # 이미지가 있는 경우 처리
        if image_path:
            image_base64 = encode_base64_content_from_file(image_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ],
            })
        else:
            # 텍스트만 있는 경우
            messages.append({"role": "user", "content": question})

        # 모델에게 메시지를 보내서 응답 받기
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
        )

        answer = response.choices[0].message.content
        return answer

    def retrieval(question, image):
        if image:
            endpoint = "retrieval/file"
            url = f"{fast_api_base}{endpoint}"

            with open(image, "rb") as f:
                files = {"file": f}
                response_retrieve = requests.post(url, params={"query": question}, files=files)
            
            image_base64 = encode_base64_content_from_file(image)
            
            if response_retrieve.status_code == 200:
                json_data = response_retrieve.json()
                retrieved_chunks = json_data.get("retrieved_chunks", [])
                combined_text = " ".join(retrieved_chunks)
                response = run_image_and_text_inference(image_base64, combined_text + question)
                return response
            else:
                response = run_image_and_text_inference(image_base64, question)
                return response
        else:
            endpoint = "retrieval"
            url = f"{fast_api_base}{endpoint}"

            response_retrieve = requests.get(url, params={"query": question})

            if response_retrieve.status_code == 200:
                json_data = response_retrieve.json()
                retrieved_chunks = json_data.get("retrieved_chunks", [])
                combined_text = " ".join(retrieved_chunks)
                response = run_text_inference(combined_text + question)
                return response
            else:
                response = run_text_inference(question)
                return response
        
    def init():
        endpoint = "init/index"
        url = f"{fast_api_base}{endpoint}"
        response = requests.post(url)
        
        if response.status_code == 200:
            return response.json().get("message", "Indexing Complete")
        else:
            return f"Indexing Failed: {response.status_code}, {response.text}"
        
    
    def generate_index(uploaded_file):
        endpoint = "indexing/file"
        url = f"{fast_api_base}{endpoint}"

        if not os.path.exists(uploaded_file.name):
            return "Indexing Failed: File not found"
        
        file_extension = os.path.splitext(uploaded_file.name)[-1].lower()
        if file_extension not in [".pdf", ".jpg", ".jpeg", ".png"]:
            return f"Indexing Failed: Unsupported file type {file_extension}"

        with open(uploaded_file.name, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)

        if response.status_code == 200:
            return response.json().get("message", "Init Complete")
        else:
            return f"Indexing Failed: {response.status_code}, {response.text}"
        
    def respond(message, chat_history):
        global flag
        # # 사용자가 아무런 입력을 하지 않은 경우 프롬프트 선택
        # if not message:
        #     message = random.choice(prompts)

        # 이미지가 있는지 확인 (리스트가 비어 있지 않은지 확인)
        image_path = None
        if isinstance(message, dict) and isinstance(message.get("files"), list) and len(message.get("files")) > 0:
            image_path = message.get("files")[0]


        # 텍스트 메시지 추출
        question = message.get("text") if isinstance(message, dict) else message

        # 사용자가 메시지를 입력했을 경우, 추가적인 프롬프트를 추가 (화면에는 나타나지 않게 처리)
        combined_question = question
        #if question:
        #   additional_prompt = random.choice(prompts)
            # 질문과 추가 프롬프트를 결합 (실제 모델에 전달할 때만 사용)
        #  combined_question += f" {additional_prompt}"

        # 인퍼런스 실행 (결합된 질문으로 실행)
        bot_message = run_inference(combined_question, chat_history, image_path)

        # 채팅 기록 업데이트 (화면에는 사용자가 입력한 원래 질문만 나타나게 함)
        if question:
            chat_history.append({"role": "user", "content": question})

        if image_path:
            flag = 1
            chat_history.append({"role": "user", "content": {"path": image_path}})

        # 모델의 응답을 채팅 기록에 추가
        chat_history.append({"role": "assistant", "content": bot_message})

        print('히스토리 내역', chat_history)
        
        return "", chat_history

    def clear_chat():
        return "", []


    with gr.Blocks(css=""" 
    #chatbot .message {
        border-radius: 12px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    #chatbot .message.user {
        background-color: gray;
        color: #0d6efd;
        text-align: left;
    }
    #chatbot .message.assistant {
        background-color: #f8f9fa;
        color: #495057;
        text-align: left;
    }
    .multimodal-textbox {
        border: 2px solid #0d6efd;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
    .btn-primary {
        background-color: #0d6efd;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        margin-top: 10px;
    }
    .btn-primary:hover {
        background-color: #084298;
        color: white;
    }
    body {
        font-family: Arial, sans-serif;
        background-color: white;
    }
""") as demo:
        gr.Markdown(
            """
            # 🏆Information Chatbot
            Ask questions
            """,
            elem_id="header",
        )
        chatbot = gr.Chatbot(type="messages")
        msg = gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple",
            placeholder="Enter message or upload file...",
            show_label=False,
        )
        clear_button = gr.Button("사진은 한 번에 하나만 올릴 수 있습니다. 사진을 추가로 업로드하고 싶다면 Clear하세요!", elem_classes="btn-primary")

        clear_button.click(fn=clear_chat, inputs=[], outputs=[msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])


        gr.Markdown("# RAG-Indexing")
        
        # file Indexing
        file_input = gr.File(label="Upload PDF or Image Files", file_types=[".pdf", ".jpg", ".jpeg", ".png"])

        with gr.Row():
            indexing_btn = gr.Button("Indexing")
            init_btn = gr.Button("Init")     
        indexing_output = gr.Textbox(label="Indexing Status", placeholder="Please provide reliable information!")
        indexing_btn.click(generate_index, inputs=[file_input], outputs=[indexing_output])
        init_btn.click(init, None, outputs=[indexing_output])

        gr.Markdown("# RAG-Retrival")
        with gr.Row():
            image_input = gr.Image(label="Upload an Image (Optional)", type="filepath")
            text_input = gr.Textbox(label="Ask a Question", placeholder="Enter your question!")
        output_box = gr.Textbox(label="Response", placeholder="Generated response from the model")

        # Trigger response generation when both inputs are provided
        retrieval_btn = gr.Button("Retrieval")
        retrieval_btn.click(retrieval, inputs=[text_input, image_input], outputs=[output_box])

    demo.queue().launch(share=False)

if __name__ == "__main__":
    main()
