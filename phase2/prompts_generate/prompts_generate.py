# ====================================
# Using Gemini
# ====================================

import os
import pandas as pd
import google.generativeai as genai
from PIL import Image
from itertools import cycle

# ================== 配置部分 ==================
# 代理设置（如果需要）
PROXY_SETTINGS = {
    "http": "http://<your ip>:<your port>",
    "https": "http://<your ip>:<your port>"
}

os.environ.update({
    'http_proxy': PROXY_SETTINGS["http"],
    'https_proxy': PROXY_SETTINGS["https"]
})

# 替换为实际图片文件夹路径和输出CSV文件路径
IMAGE_FOLDER = ""
OUTPUT_CSV_FILE = ""

# API 密钥池，用以轮询
API_KEYS = [

]

key_iterator = cycle(API_KEYS)
current_key = next(key_iterator)
genai.configure(api_key=current_key)

if not API_KEYS:
    raise ValueError("API_KEYS 列表不能为空，请至少提供一个有效的密钥。")

# ================== 主体部分 ==================

def switch_to_next_key():
    global current_key
    try:
        current_key = next(key_iterator)
        genai.configure(api_key=current_key)
        print(f"切换到下一个 API 密钥。当前密钥: {current_key[:9]}...")
    except StopIteration:
        raise RuntimeError("所有 API 密钥都已遍历完毕，且均不可用。")

def generate_prompts_for_image(image_path, num_prompts=8):
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"警告: 文件 {image_path} 不存在，跳过。")
        return ["File not found"] * num_prompts

    prompt_text = f"""
    You are an expert drone video script assistant for a world model that generates videos. Given a single drone shot frame, your task is to generate {num_prompts} distinct, high-quality, and structured video generation prompts. These prompts will serve as inputs to a world model.

    The prompts should adhere to a specific format and include the following elements:
    
    1.  **Drone's action**: Describe the drone's movement (e.g., forward, backward, left, right, up, down, rotation).
    2.  **Gimbal/Camera adjustment**: Detail how the camera's gimbal is adjusted (e.g., tilting up/down) and the camera's perspective (e.g., first-person, top-down).
    3.  **Scene content**: Briefly describe the main scene, objects, or people in the frame.
    4.  **Action intent**: Explain the purpose of the drone's movement, such as tracking a target or capturing a specific view.
    
    It is crucial that the {num_prompts} prompts are distinct from each other, but not overly complex or dramatically different. They should vary in:
    -   **Action combinations**: Mix different movements (e.g., forward while rotating, ascending while tilting down).
    -   **Sequence/Timing**: Describe actions happening in a specific order.
    -   **Magnitude/Speed**: Use words like "slowly," "rapidly," "slightly," or "gradually."
    -   **Intent**: Change the focus or purpose of the shot slightly.

    Please ensure the prompts are structured and in English, similar to these examples:
    -   The drone slowly rotates to the right while adjusting the camera gimbal downwards to keep the group of people hiking on the mountainside trail in the center of the view. The hikers are moving along the narrow path carved into the rock face.
    -   The video is egocentric/first-person perspective, captured from a camera mounted on a drone. The drone descended vertically while tilting its camera gimbal upward slightly and ascended, capturing a top-down view of a rocky area with scattered structures, trees, and a blue dome-like object, reaching a stable hovering position at the end.
    -   The video is egocentric/first-person perspective, captured from a camera mounted on a drone. The drone moves forward while the camera gimbal adjusts to track the motion of a white car traveling along the road in a forward direction, ultimately reaching an intersection where the car turns to the left.

    The generated prompts should be clear and concise inputs for a video generation model.
    """

    for _ in range(len(API_KEYS)):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content([prompt_text, img],
                                                generation_config=genai.types.GenerationConfig(
                                                    candidate_count=num_prompts,
                                                ))
            prompts = [candidate.content.parts[0].text for candidate in response.candidates]
            return prompts
        except genai.types.generation_types.BlockedPromptException as e:
            print(f"警告: 提示被拦截。图片: {os.path.basename(image_path)}, 错误: {e}")
            return ["Blocked content"] * num_prompts
        except Exception as e:
            print(f"使用密钥 {current_key[:5]}... 处理图片 {os.path.basename(image_path)} 时出错: {e}")
            switch_to_next_key()

    raise RuntimeError("所有 API 密钥均已尝试，但均无法成功调用 API。")

def process_images_in_folder():
    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError(f"错误: 文件夹 '{IMAGE_FOLDER}' 不存在。")

    data_to_save = []
    
    for filename in sorted(os.listdir(IMAGE_FOLDER)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(IMAGE_FOLDER, filename)
            print(f"正在处理图片: {filename}...")
            
            try:
                prompts = generate_prompts_for_image(image_path, num_prompts=8)
                row = [filename] + prompts
                data_to_save.append(row)
            except RuntimeError as e:
                print(e)
                break

    columns = ["frame_name"] + [f"prompt_{i+1}" for i in range(8)]
    df = pd.DataFrame(data_to_save, columns=columns)
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
    
    print(f"所有图片处理完毕。结果已保存到 '{OUTPUT_CSV_FILE}'。")

if __name__ == "__main__":
    process_images_in_folder()


# ====================================
# Using GPT
# ====================================

import os
import sys
import base64
import pandas as pd
from openai import AzureOpenAI

# ================== 配置部分 ==================

# 代理设置（如果需要）
PROXY_SETTINGS = {
    "http": "http://<your ip>:<your port>",
    "https": "http://<your ip>:<your port>"
}

os.environ.update({
    'http_proxy': PROXY_SETTINGS["http"],
    'https_proxy': PROXY_SETTINGS["https"]
})

# Azure OpenAI 配置
AZURE_API_KEY = "xxxx"
AZURE_API_VERSION = "2024-07-01-preview"
AZURE_ENDPOINT = "xxx"
MODEL_NAME = "gpt-4o"

# 图片文件夹 & 输出路径
IMAGE_FOLDER = ""
OUTPUT_CSV_FILE = ""

# 每张图片生成几个 prompt
NUM_PROMPTS = 8


# 初始化 Azure OpenAI 客户端
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

# ================== 主体部分 ==================

def generate_prompts_for_image(image_path, num_prompts=8):
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"警告: 文件 {image_path} 不存在，跳过。")
        return ["File not found"] * num_prompts

    prompt_text = f"""
    You are an expert drone video script assistant for a world model that generates videos.
    Given a single drone shot frame, your task is to generate {num_prompts} distinct,
    and well-structured video generation prompts of high quality.

    Specifically, these batch of frames are all sampled at night scenarios, be careful when analyzing them because of the dark environment

    The prompts should adhere to a specific format and include:
    1. Drone's action (movement direction, 6 degrees of translation freedom, and 2 degrees of rotation freedom)
    2. Gimbal/Camera adjustment (fixed angle, tilt up/down, perspective)
    3. Consice scene content (main objects, people, or landscape)
    4. Action intent (not necessary, generally is tracking/orbiting or just moves without intents)

    The basic motion trajectories should be the same among the 8 prompts(like all should be forward and rotate right, differs in the aspects below)
    Only make slight variations across:
    - Magnitude/Degree/Speed/Time of motions between different prompts


    Examples(3 different videos, not the same one):
    - The drone slowly rotates to the right while adjusting the camera gimbal downwards to keep the group of people hiking on the mountainside trail in the center of the view. The hikers are moving along the narrow path carved into the rock face.
    - The video is egocentric/first-person perspective, captured from a camera mounted on a drone. The drone descended vertically while tilting its camera gimbal upward slightly and ascended, capturing a top-down view of a rocky area with scattered structures, trees, and a blue dome-like object, reaching a stable hovering position at the end.
    - The video is egocentric/first-person perspective, captured from a camera mounted on a drone. The drone moves forward while the camera gimbal adjusts to track the motion of a white car traveling along the road in a forward direction, ultimately reaching an intersection where the car turns to the left.

    Output each prompt on a separate line. 
    Without number like 1., 2.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            }
        ]
    )

    output_text = response.choices[0].message.content
    prompts = [line.strip("- ").strip() for line in output_text.split("\n") if line.strip()]
    return prompts[:num_prompts] + [""] * (num_prompts - len(prompts))


def process_images_in_folder():
    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError(f"错误: 文件夹 '{IMAGE_FOLDER}' 不存在。")

    data_to_save = []
    
    for filename in sorted(os.listdir(IMAGE_FOLDER)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(IMAGE_FOLDER, filename)
            print(f"正在处理图片: {filename}...")
            
            try:
                prompts = generate_prompts_for_image(image_path, num_prompts=NUM_PROMPTS)
                row = [filename] + prompts
                data_to_save.append(row)
            except Exception as e:
                print(f"处理图片 {filename} 时出错: {e}")
                break

    columns = ["frame_name"] + [f"prompt_{i+1}" for i in range(NUM_PROMPTS)]
    df = pd.DataFrame(data_to_save, columns=columns)
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
    
    print(f"所有图片处理完毕。结果已保存到 '{OUTPUT_CSV_FILE}'。")


if __name__ == "__main__":
    process_images_in_folder()
    sys.exit(0)
