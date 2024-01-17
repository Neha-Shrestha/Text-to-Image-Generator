import gradio as gr
import kruto_module

def text_to_image(text):
    generated_image = kruto_module.generate_image(text)
    return generated_image

iface = gr.Interface(
    fn=text_to_image,
    inputs=gr.Textbox(lines=1, label="Enter text here"),
    outputs="image",
    title="Text-to-Image Generator",
    description="This Text-to-Image Generates face given a text input of personalities.",
    theme="default",
    allow_flagging=False,
)

if __name__ == "__main__":
    iface.launch()