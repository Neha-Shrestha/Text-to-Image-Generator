import gradio as gr
from ldm import generate_image
from trainer import train, cancel
from testing import compute_fid

with gr.Blocks(title="Text-to-Image Generator") as app:
    gr.Markdown("# Text-to-Image Generator: 7th Semester Project")
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
                
            with gr.Column():
                gr.Image(label="Model Loss", value="./images/final.png", width=500)
            
        with gr.Row():
            with gr.Column():
                inference_output = gr.Image(label="Timesteps")
                generate_btn.click(
                    generate_image,
                    inputs=[text], 
                    outputs=[image_output, inference_output],
                )

    with gr.Tab("Training"):
        gr.Markdown("### Train a Text-to-Image Generator by inserting following Hyperparameters.")
        with gr.Row():
            epochs = gr.Textbox(label="Epochs", value=1000)
            batch_size = gr.Textbox(label="Batch Size", value=20)
            lr = gr.Textbox(label="Learning Rate", value=0.001)
        
        with gr.Row():
            loss_fn = gr.Radio(["MSE", "MAE"], label="Loss Function", value="MSE")
            optimizer = gr.Radio(["Adam", "SGD"], label="Optimizer", value="Adam")
            latent_folder = gr.Textbox(label="Latent Folder Location", value="./data/face/images")

        with gr.Row():
            preprocess = gr.Radio(["Yes", "No"], label="Preprocessing", value="No")
            data_folder = gr.Textbox(label="Data Folder Location")
        
        with gr.Row():
            train_btn = gr.Button("Train")
            cancel_btn = gr.Button("Cancel")
        
        progress_output = gr.Textbox()
        
        train_btn.click(
            train, 
            inputs=[epochs, batch_size, lr, loss_fn, optimizer, latent_folder, preprocess, data_folder],
            outputs=progress_output
        )
        cancel_btn.click(cancel)

if __name__ == "__main__":
    app.launch()