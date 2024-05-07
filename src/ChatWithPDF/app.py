# This file is the UI for the chatwithpdf
import gradio as gr
from core import process_pdf,process_query
def process_pdf_and_text(pdf_file_path, user_text):
    print(f"[INFO] The pdf file is in the {pdf_file_path}")
    if not hasattr(process_pdf_and_text,"_called"):
        process_pdf(pdf_file_path)
        process_pdf_and_text._called = True
    
    result = process_query(user_text)
    return result

def main():
    # input components
    pdf_input = gr.File(label="Upload PDF File")
    text_input = gr.TextArea(label="Enter the query")
    # output component
    output_text = gr.TextArea()
    
    # app interface
    demo = gr.Interface(
        fn=process_pdf_and_text,
        inputs=[pdf_input, text_input],
        outputs=output_text,
        title="Chat With PDF",
        description="RAG based Chat with pdf"
    )

    return demo
    
if __name__ == "__main__":
    demo = main()
    demo.launch()