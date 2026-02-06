import base64
import copy
import inspect
import json
import os
import shutil

import gradio as gr
import yaml
from datasets import Dataset

from data_juicer.ops.base_op import OPERATORS
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import SpecialTokens, remove_special_tokens

demo_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(demo_path))


# 图片本地路径转换为 base64 格式
def covert_image_to_base64(image_path):
    # 获得文件后缀名
    ext = image_path.split(".")[-1]
    if ext not in ["gif", "jpeg", "png"]:
        ext = "jpeg"

    with open(image_path, "rb") as image_file:
        # Read the file
        encoded_string = base64.b64encode(image_file.read())

        # Convert bytes to string
        base64_data = encoded_string.decode("utf-8")

        # 生成base64编码的地址
        base64_url = f"data:image/{ext};base64,{base64_data}"
        return base64_url


def format_cover_html(project_img_path):
    readme_link = 'https://github.com/datajuicer/data-juicer'
    config = {
        'name': "Data-Juicer",
        'label': "Op Insight",
        'description': f'A One-Stop Data Processing System for Large Language Models.',
        'introduction': 
        "This project is being actively updated and maintained, and we will periodically enhance and add more features and data recipes. <br>"
        "We welcome you to join us in promoting LLM data development and research!<br>",
        'demo':"You can experience the effect of the operators of Data-Juicer",
        'note':'Note: Due to resource limitations, only a subset of operators is available here. see more details in <a href="{readme_link}">GitHub</a>'
    }
    # image_src = covert_image_to_base64(project_img_path)
    # <div class="project_img"> <img src={image_src} /> </div>
    # <div class='project_cover'>
    return f"""
    <div>
    <div class="project_name">{config.get("name", "")} </div>
    <div class="project_desc">{config.get("description", "")}</div>
    <div class="project_desc">{config.get("introduction", "")}</div>
    <div class="project_desc">{config.get("demo", "")}</div>
    <div class="project_desc">{config.get("note", "")}</div>
</div>
"""
op_text = ''
docs_file = os.path.join(project_path, 'docs/Operators.md')
if os.path.exists(docs_file):
    with open(os.path.join(project_path, 'docs/Operators.md'), 'r') as f:
        op_text = f.read()

def extract_op_desc(markdown_text,  header):
    start_index = markdown_text.find(header)
    end_index = markdown_text.find("\n##", start_index + len(header)) 
    return markdown_text[start_index+ len(header):end_index].strip()

op_desc = f"<div style='text-align: center;'>{extract_op_desc(op_text, '## Overview').split('All the specific ')[0].strip()}</div>"
op_list_desc = {
    'mapper':extract_op_desc(op_text, '## Mapper <a name="mapper"/>'),
    'filter':extract_op_desc(op_text, '## Filter <a name="filter"/>'),
    'deduplicator':extract_op_desc(op_text, '## Deduplicator <a name="deduplicator"/>'),
    'selector':extract_op_desc(op_text, '## Selector <a name="selector"/>'),
}

op_types = ['mapper', 'filter', 'deduplicator']
local_ops_dict = {op_type:[] for op_type in op_types}
multimodal = os.getenv('MULTI_MODAL', True)
multimodal_visible = False
cache_dir = './cache'
text_key = 'text'
image_key = 'images'
audio_key = 'audios'
video_key = 'videos'

def get_op_lists(op_type):
    use_local_op = os.getenv('USE_LOCAL_OP', False)
    if not use_local_op:
        all_ops = list(OPERATORS.modules.keys())
        options = [
            name for name in all_ops if name.endswith(op_type)
        ]
    else:
        options = local_ops_dict.get(op_type, [])

    for exclude in ['image', 'video', 'audio']:
        options = [name for name in options if multimodal or exclude not in name]
    return options

def show_code(op_name):
    op_class = OPERATORS.modules[op_name]
    text = inspect.getsourcelines(op_class)

    init_signature = inspect.signature(op_class.__init__)

    # 输出每个参数的名字和默认值
    default_params = dict()
    for name, parameter in init_signature.parameters.items():
        if name in ['self', 'args', 'kwargs']:
            continue  # 跳过 'self' 参数
        if parameter.default is not inspect.Parameter.empty:
            default_params[name] = parameter.default

    return ''.join(text[0]), yaml.dump(default_params)

def change_visible(op_name, show_text):
    text_visible = show_text
    video_visible = False
    audio_visible = False
    image_visible = False
    if 'video' in op_name:
        video_visible = True
    elif 'audio' in op_name:
        audio_visible = True
    elif 'image' in op_name:
        image_visible = True
    elif 'document' in op_name:
        text_visible = True
    return gr.update(visible=text_visible), gr.update(visible=image_visible), gr.update(visible=video_visible), gr.update(visible=audio_visible),  gr.update(visible=text_visible), gr.update(visible=image_visible), gr.update(visible=video_visible), gr.update(visible=audio_visible)


def clear_directory(directory=cache_dir):
    for item in os.listdir(directory):
        if item == '.gitkeep':
            continue
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)  # 删除文件或链接
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # 递归删除目录


def copy_func(file):
    cache_file = None
    if file:
        filename= os.path.basename(file)
        cache_file = os.path.join(cache_dir, filename)
        shutil.copyfile(file, cache_file)
    return cache_file


def encode_sample(input_text, input_image, input_video, input_audio, is_batched_op=False):
    sample = dict()
    sample[image_key]= [input_image] if input_image else []
    sample[video_key]=[input_video] if input_video else []
    sample[audio_key]=[input_audio] if input_audio else []

    if input_image:
        input_text += SpecialTokens.image
    if input_video:
        input_text += SpecialTokens.video
    if input_audio:
        input_text += SpecialTokens.audio
    sample[text_key]=input_text

    if is_batched_op:
        for k, v in sample.items():
            sample[k] = [v]
    return sample


def decode_sample(output_sample, is_batched_op=False):
    if is_batched_op:
        for k, v in output_sample.items():
            output_sample[k] = v[-1]

    output_text = remove_special_tokens(output_sample[text_key])
    output_image = output_sample[image_key][0] if output_sample[image_key] else None
    output_video = output_sample[video_key][0] if output_sample[video_key] else None 
    output_audio = output_sample[audio_key][0] if output_sample[audio_key] else None
    image_file = copy_func(output_image)
    video_file = copy_func(output_video)
    audio_file = copy_func(output_audio)
    return output_text, image_file, video_file, audio_file


def create_tab_layout(op_tab, op_type, run_op, has_stats=False):
    with op_tab:
        options = get_op_lists(op_type)
        label = f'Select a {op_type} to show details'
        with gr.Row():
            op_selector = gr.Dropdown(value=options[0], label=label, choices=options, interactive=True)
            with gr.Column():
                gr.Markdown(" **Op Parameters**")
                op_params = gr.Code(label="Yaml",language='yaml', interactive=True)
            run_button = gr.Button(value="🚀Run")
            show_code_button = gr.Button(value="🔍Show Code")
        show_text = gr.Checkbox(value=True,visible=False)
        with gr.Column():
            with gr.Group('Inputs'):
                gr.Markdown(" **Inputs**")
                with gr.Row():
                    input_text = gr.TextArea(label="Text",interactive=True,scale=2)
                    input_image = gr.Image(label='Image', type='filepath', visible=multimodal_visible)
                    input_video = gr.Video(label='Video', visible=multimodal_visible)
                    input_audio = gr.Audio(label='Audio', type='filepath', visible=multimodal_visible)

            with gr.Group('Outputs'):
                gr.Markdown(" **Outputs**")
                with gr.Row():
                    output_text = gr.TextArea(label="Text",interactive=False,scale=2)
                    output_image = gr.Image(label='Image', type='filepath', visible=multimodal_visible)
                    output_video = gr.Video(label='Video', visible=multimodal_visible,)
                    output_audio = gr.Audio(label='Audio', type='filepath', visible=multimodal_visible)

                with gr.Row():
                    if has_stats:
                        output_stats = gr.Json(label='Stats')
                        output_keep = gr.Text(label='Keep or not?', interactive=False)

            code = gr.Code(label='Source', language='python')
        inputs = [input_text, input_image, input_video, input_audio, op_selector, op_params]
        outputs = [output_text, output_image, output_video, output_audio]
        if has_stats:
            outputs.append(output_stats)
            outputs.append(output_keep)

        def run_func(*args):
            try:
                try:
                    args = list(args)
                    op_params = args.pop()
                    params = yaml.safe_load(op_params)
                except:
                    params = {}
                if params is None:
                    params = {}
                return run_op(*args, params)
            except Exception as e:
                gr.Error(str(e))
                print(e)
                return outputs

        show_code_button.click(show_code, inputs=[op_selector], outputs=[code, op_params])
        show_code_button.click(change_visible, inputs=[op_selector,show_text], outputs=outputs[:4] + inputs[:4])        
        run_button.click(run_func, inputs=inputs, outputs=outputs)
        run_button.click(change_visible, inputs=[op_selector,show_text], outputs=outputs[:4] + inputs[:4])   
        op_selector.select(show_code, inputs=[op_selector], outputs=[code, op_params])
        op_selector.select(change_visible, inputs=[op_selector,show_text], outputs=outputs[:4] + inputs[:4])
        op_tab.select(change_visible, inputs=[op_selector,show_text], outputs=outputs[:4] + inputs[:4])
        op_tab.select(show_code, inputs=[op_selector], outputs=[code, op_params])

def create_mapper_tab(op_type, op_tab):
    with op_tab:
        def run_op(input_text, input_image, input_video, input_audio, op_name, op_params):
            op_class = OPERATORS.modules[op_name]
            op = op_class(**op_params)
            is_batched_op = op.is_batched_op()
            sample = encode_sample(input_text, input_image, input_video, input_audio, is_batched_op)
            output_sample = op.process(copy.deepcopy(sample))
            return decode_sample(output_sample, is_batched_op)
        create_tab_layout(op_tab, op_type, run_op)


def create_filter_tab(op_type, op_tab):
    def run_op(input_text, input_image, input_video, input_audio, op_name, op_params):
        op_class = OPERATORS.modules[op_name]
        op = op_class(**op_params)
        sample = encode_sample(input_text, input_image, input_video, input_audio)
        sample[Fields.stats] = dict()
        output_sample = op.compute_stats(copy.deepcopy(sample))
        if op.process(output_sample):
            output_keep = 'Yes'
        else:
            output_keep = 'No'
        output_stats = output_sample[Fields.stats]
        return *decode_sample(output_sample), output_stats, output_keep
    create_tab_layout(op_tab, op_type, run_op, has_stats=True)


def create_deduplicator_tab(op_type, op_tab):
    with op_tab:
        def run_op(input_text, input_image, input_video, input_audio, input_text2, input_image2, input_video2, input_audio2, op_name, op_params):
            op_class = OPERATORS.modules[op_name]
            op = op_class(**op_params)
            sample = encode_sample(input_text, input_image, input_video, input_audio)
            sample2 = encode_sample(input_text2, input_image2, input_video2, input_audio2)
            output_sample = op.compute_hash(copy.deepcopy(sample))
            output_sample2 = op.compute_hash(copy.deepcopy(sample2))
            ds = Dataset.from_list([output_sample, output_sample2])
            hash_values = ds.remove_columns([text_key, image_key, video_key, audio_key]).to_dict()
            ds.cleanup_cache_files()
            for key, values in hash_values.items():
                new_values = []
                for value in values:
                    if isinstance(value, list):
                        new_values.append([v.hex() for v in value])
                hash_values[key] = new_values or values
            _, dedup_pairs = op.process(ds, show_num=1)
            if dedup_pairs:
                dedup = "Yes"
            else:
                dedup = "No"
            
            return json.dumps(hash_values), dedup
        create_tab_double_layout(op_tab, op_type, run_op)


def create_tab_double_layout(op_tab, op_type, run_op):
    with op_tab:
        options = get_op_lists(op_type)
        label = f'Select a {op_type} to show details'
        with gr.Row():
            op_selector = gr.Dropdown(value=options[0], label=label, choices=options, interactive=True)
            with gr.Column():
                gr.Markdown(" **Op Parameters**")
                op_params = gr.Code(label="Yaml",language='yaml', interactive=True)
            run_button = gr.Button(value="🚀Run")
            show_code_button = gr.Button(value="🔍Show Code")
        show_text = gr.Checkbox(value=False,visible=False)
        with gr.Column():
            with gr.Group('Inputs'):
                gr.Markdown(" **Inputs**")
                with gr.Row():
                    
                    input_text = gr.TextArea(label="Text",interactive=True,)
                    input_text2 = gr.TextArea(label="Text",interactive=True,)
                    input_image = gr.Image(label='Image', type='filepath', visible=multimodal_visible)
                    input_image2 = gr.Image(label='Image', type='filepath', visible=multimodal_visible)
                    input_video = gr.Video(label='Video', visible=multimodal_visible)
                    input_video2 = gr.Video(label='Video', visible=multimodal_visible)
                    input_audio = gr.Audio(label='Audio', type='filepath', visible=multimodal_visible)
                    input_audio2 = gr.Audio(label='Audio', type='filepath', visible=multimodal_visible)

            with gr.Group('Outputs'):
                gr.Markdown(" **Outputs**")
                with gr.Row():
                    output_deduplicated_pairs = gr.Json(label='Deduplicated pairs')
                    output_deduplicated = gr.Text(label='Deduplicate or not?', interactive=False)
                    
            code = gr.Code(label='Source', language='python')
        inputs = [input_text, input_image, input_video, input_audio, input_text2, input_image2, input_video2, input_audio2, op_selector, op_params]
        outputs = [output_deduplicated_pairs, output_deduplicated]

        def run_func(*args):
            try:
                try:
                    args = list(args)
                    op_params = args.pop()
                    params = yaml.safe_load(op_params)
                except:
                    params = {}
                if params is None:
                    params = {}
                return run_op(*args, params)
            except Exception as e:
                gr.Error(str(e))
                print(e)
                return outputs

        show_code_button.click(show_code, inputs=[op_selector], outputs=[code, op_params])
        show_code_button.click(change_visible, inputs=[op_selector, show_text], outputs=inputs[:8])        
        run_button.click(run_func, inputs=inputs, outputs=outputs)
        run_button.click(change_visible, inputs=[op_selector,show_text], outputs=inputs[:8])   
        op_selector.select(show_code, inputs=[op_selector], outputs=[code, op_params])
        op_selector.select(change_visible, inputs=[op_selector,show_text], outputs=inputs[:8])
        op_tab.select(change_visible, inputs=[op_selector,show_text], outputs= inputs[:8])
        op_tab.select(show_code, inputs=[op_selector], outputs=[code, op_params])

with gr.Blocks(css="./app.css") as demo:
    dj_image = os.path.join(project_path, 'docs/imgs/data-juicer.jpg')
    gr.HTML(format_cover_html(dj_image))
    
    with gr.Accordion(label='Op Insight',open=True):
        tabs = gr.Tabs()
        with tabs:
            op_tabs = {op_type: gr.Tab(label=op_type.capitalize() + 's') for op_type in op_types}
            for op_type, op_tab in op_tabs.items():
                    create_op_tab_func = globals().get(f'create_{op_type}_tab', None)
                    if callable(create_op_tab_func):       
                        create_op_tab_func(op_type, op_tab)
                    else:
                        gr.Error(f'{op_type} not callable')
    demo.load(clear_directory, every=10)
    demo.launch()
