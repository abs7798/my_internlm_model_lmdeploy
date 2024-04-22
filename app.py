import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig
import os

# download internlm2 to the base_path directory using git tool
base_path = './internlm2-chat-7b'
os.system(f'mkdir internlm2-chat-7b')

print(f'创建文件夹{os.path.isdir(base_path)}')
os.system(f'cd {base_path}')

os.system("git lfs install")
os.system(f'git clone https://code.openxlab.org.cn/abs7798/my_internlm_model.git')
os.system(f'cd {base_path} && git lfs pull')
os.system("pip install sentencepiece")
os.system("pip install einops")

backend_config = TurbomindEngineConfig(session_len=8192) # 图片分辨率较高时请调高session_len
# pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
pipe = pipeline(base_path, backend_config=backend_config)

def model(image, text):
    if image is None:
        return [(text, "请上传一张图片。")]
    else:
        response = pipe((text, image)).text
        return [(text, response)]

demo = gr.Interface(fn=model, inputs=[gr.Image(type="pil"), gr.Textbox()], outputs=gr.Chatbot())
demo.launch()   