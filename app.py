import gradio as gr
from orchestrator import orch

def main():
    chat = gr.ChatInterface(fn=orch, type="messages")
    chat.launch(inbrowser = True)

if __name__ == "__main__":
    main()
