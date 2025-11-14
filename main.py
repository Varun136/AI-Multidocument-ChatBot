import os
import gradio as gr
from dotenv import load_dotenv
from agent import DocumentEvaluationAgent

load_dotenv(override=True)


if __name__ == "__main__":
    collection_name = "collection_1"
    document_evaluation_agent = DocumentEvaluationAgent(collection_name)

    def upload_file(file):
        uploaded_file_path = file.name
        return document_evaluation_agent.initiate_agent(uploaded_file_path)

    def chat_with_doc(user_message, chat_history):
        response = document_evaluation_agent.invoke(user_message)
        return response 

    with gr.Blocks(title="QA Chatbot") as demo:
        gr.Markdown("Ask Questions About Your File\nUpload a file and start chatting!")

        with gr.Row():
            file_input = gr.File(label="Upload file (PDF, DOCX, XLSX, or Image)")
            upload_status = gr.Textbox(label="Upload status")

        file_input.upload(upload_file, inputs=file_input, outputs=upload_status)

        chatbot = gr.ChatInterface(
            fn=chat_with_doc,
            title="Chat with your document",
            description="Ask any question about the uploaded file",
        )

        demo.launch()
