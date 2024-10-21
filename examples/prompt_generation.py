import csv
from datasets import Dataset
import requests
from typing import List, Dict, Tuple

# 设置DeepSeek API密钥
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

def load_csv_to_dataset(file_path: str) -> Dataset:
    dataset = Dataset.from_csv(file_path)
    return dataset

def generate_prompt(task: str) -> str:
    base_prompt = """我正在编写一本小说，小说的场景需要一幅配图。请根据我提供的图片描述生成一条用于文生图模型的提示词。提示词应具备清晰的结构，并对图片的主体和细节进行适当的扩展和优化。以下是具体的要求：

1、主体提取：请明确指出图片的主要人物或元素，如场景中的人数、环境等。
2、敏感词替换：请识别图片描述中的敏感词汇或不适宜直接表达的部分，用更具风格化且符合上下文的描述进行替换。在替换时，保留原意并以更艺术的形式呈现，同时避免使用不合适的词语。
3、内容丰富与细节增强：请对描述内容进行适当的扩展，增加场景中的细节，如天气、环境色彩、人物的表情和姿态等，使画面更具层次感和氛围感。
4、风格和质量提示：请在提示词中增加影响图像质量和风格的提示，如光照、构图、色调等，使其更符合小说的氛围和叙述风格。
5、非写实风格与动画风格：请确保输出的图像尽量采用非写实的表达方式，可以指定某种动画风格（如手绘、卡通、幻想风格等），使图像更具创意性和艺术感，契合小说的独特风格。
6、中英文提示词生成：最终请分别生成中文和英文的提示词，确保描述一致，适合不同语言场景下的配图生成需求。
"""
    return f"{base_prompt}\n\n图片描述: {task}"

def call_deepseek_api(prompt: str) -> Tuple[str, str]:
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        model_output = response.json()["choices"][0]["message"]["content"].strip()
        parts = model_output.split("|")
        prompt_zh = parts[0] if len(parts) >= 1 else ""
        prompt_en = parts[1] if len(parts) >= 2 else ""
    else:
        prompt_zh = prompt_en = "API调用失败"
    
    return prompt_zh, prompt_en

def process_dataset(dataset: Dataset) -> List[Dict]:
    results = []
    for row in dataset:
        task_id = row["task_id"]
        task = row["task"]
        prompt = generate_prompt(task)
        prompt_zh, prompt_en = call_deepseek_api(prompt)
        results.append({
            "task_id": task_id,
            "task": task,
            "prompt_zh": prompt_zh,
            "prompt_en": prompt_en
        })
        print(f"{task_id},{task},{prompt_zh}|{prompt_en}")
    return results

def save_results_to_csv(results: List[Dict], output_file: str):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "task", "prompt_zh", "prompt_en"])
        writer.writeheader()
        writer.writerows(results)

def main():
    input_file = "input.csv"
    output_file = "output.csv"
    
    dataset = load_csv_to_dataset(input_file)
    results = process_dataset(dataset)
    save_results_to_csv(results, output_file)
    
    print(f"处理完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
