import gradio as gr
from orchestrator import orch

def main():
    chat = gr.ChatInterface(fn=orch)
    chat.launch(inbrowser = True)

if __name__ == "__main__":
    main()
