import gradio as gr

def greet(name):
    with gr.Row():
        input_img = gr.Image(label="Input")
        mask_img = gr.Image(label="Mas")
        output_image = gr.Image(label="Output")
        
    with gr.Block():
        prompt_text = gr.Textbox(lines=1, label="Prompt")
        
    with gr.Row():
        submit = gr.Button("Submit")
    return "Hello, " + name + "!"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()