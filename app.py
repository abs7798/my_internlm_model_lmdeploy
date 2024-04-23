import gradio as gr
import os

# download internlm2 to the base_path directory using git tool
base_path = './my_internlm_model'
os.system(f'mkdir my_internlm_model')

print(f'创建文件夹{os.path.isdir(base_path)}')
os.system(f'cd {base_path}')
os.system("git lfs install")
os.system(f'git clone https://code.openxlab.org.cn/abs7798/my_internlm_model.git')
os.system(f'cd {base_path} && git lfs pull')
os.system("pip install sentencepiece")
os.system("pip install einops")
os.system("pip install lmdeploy[all]==0.3.0")

import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig
backend_config = TurbomindEngineConfig(session_len=8192,cache_max_entry_count=0.7) # 图片分辨率较高时请调高session_len
pipe = pipeline(base_path, model_name='my_model', backend_config=backend_config)

#'''text= '你是谁'
#response = pipe([text])
#print(response[0].text)'''

def model(text):
    response = pipe([text])
    print(response)
    res = response[0].text
    return [(text, res)]

demo = gr.Interface(fn=model, inputs= gr.Textbox(), outputs=gr.Chatbot())
demo.launch()