import gradio as gr
import ldm_module

iface = gr.Interface(
    fn=ldm_module.generate_image,
    inputs=gr.Textbox(lines=1, label="Enter text here"),
    outputs="image",
    title="Text-to-Image Generator",
    description="Provide name and generate face.",
    theme="default",
    allow_flagging=False,
    examples=[
        ["Barack Obama"],
        ["Cristiano Ronaldo"],
        ["Donald Trump"]
    ]
)

if __name__ == "__main__":
    iface.launch()
