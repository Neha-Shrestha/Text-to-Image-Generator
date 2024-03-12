import gradio as gr
from ldm import generate_image
from trainer import train

with gr.Blocks(title="Text-to-Image Generator") as app:
    gr.Markdown("# Text-to-Image Generator: 7th Semester Project")
    with gr.Tab("Training"):
        gr.Markdown("### Train a Text-to-Image Generator by inserting following Hyperparameters.")
        with gr.Row():
            with gr.Column(scale=5):
                epochs = gr.Textbox(label="Epochs")
                lr = gr.Textbox(label="Learning Rate")
                loss_fn = gr.Radio(["MSELoss", "MAELoss"], label="Loss Function")
                optimizer = gr.Radio(["Adam", "SGD"], label="Optimizer")
                train_btn = gr.Button("Train")
                
            with gr.Column(scale=5):
                training_loss = gr.Label(label="Training Loss Value")
                training_image = gr.Image(label="Training Generated Image")
                train_btn.click(
                    train, 
                    inputs=[epochs, lr, loss_fn, optimizer], 
                    outputs=[training_loss, training_image]
                )
        
    with gr.Tab("Inference"):
        gr.Markdown("### Insert a text and generate a face.")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Enter text prompt")
                gr.Examples([
                    ["Barack Obama"], 
                    ["Cristiano Ronaldo"], 
                    ["Donald Trump"]
                ], inputs=[text])
                generate_btn = gr.Button("Generate")
                gr.ClearButton([text])
            
            with gr.Column():
                image_output = gr.Image(label="Generated image", width=500)
                generate_btn.click(
                    generate_image,
                    inputs=[text], 
                    outputs=[image_output],
                )      
        
            with gr.Column():
                gr.Image(label="Model Loss", value="./images/final.png", width=500)

if __name__ == "__main__":
    app.launch()

# import gradio as gr
# import ldm

# iface = gr.Interface(
#     fn=ldm.generate_image,
#     inputs=gr.Textbox(lines=1, label="Enter text here"),
#     outputs="image",
#     title="Text-to-Image Generator",
#     description="Provide name and generate face.",
#     theme="default",
#     allow_flagging=False,
#     examples=[
#         ["Barack Obama"],
#         ["Cristiano Ronaldo"],
#         ["Donald Trump"]
#     ]
# )

# if __name__ == "__main__":
#     iface.launch()