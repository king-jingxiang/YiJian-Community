import csv
from datasets import Dataset
import requests
from typing import List, Dict, Tuple

import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler('prompt_generation_logs.md', encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )

import os
# 设置DeepSeek API密钥
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"


def load_csv_to_dataset(file_path: str) -> Dataset:
    dataset = Dataset.from_csv(file_path)
    return dataset


def generate_prompt(task: str) -> str:
    base_prompt = """xxxx"""
    return f"{base_prompt}\n\n图片描述: {task}"


def extract_prompt(data):
    prompt_zh = ""
    prompt_en = ""
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"这里的大语言模型生成的提示词，请你去除不相关的，把最终的中文提示词和英文提示词输出，使用|进行分割，方便我后续进行处理 \n提示词描述：{data}",
            },
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
        print(f"{model_output}")
        logging.info(f"{model_output}")
        parts = model_output.split("|")
        prompt_zh = parts[0] if len(parts) >= 1 else ""
        prompt_en = parts[1] if len(parts) >= 2 else ""
    else:
        prompt_zh = prompt_en = "API调用失败"
    return prompt_zh, prompt_en


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

        #  调用大语言模型提取最终生成的提示词
        print(f"{model_output}")
        logging.info(f"\n{model_output}")
        prompt_zh, prompt_en = extract_prompt(model_output)
    else:
        prompt_zh = prompt_en = "API调用失败"

    return prompt_zh, prompt_en


def process_dataset(dataset: Dataset) -> List[Dict]:
    results = []
    for row in dataset:
        task_id = row["task_id"]
        task = row["task"]
        prompt = generate_prompt(task)
        print("-" * 20)
        print(f"{task_id},{task}\n")
        logging.info("-" * 20)
        logging.info(f"{task_id},{task}\n")
        prompt_zh, prompt_en = call_deepseek_api(prompt)
        results.append(
            {
                "task_id": task_id,
                "task": task,
                "prompt_zh": prompt_zh,
                "prompt_en": prompt_en,
            }
        )
        print(f"{task_id},{task},{prompt_zh}|{prompt_en}")
    return results


def save_results_to_csv(results: List[Dict], output_file: str):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task_id", "task", "prompt_zh", "prompt_en"]
        )
        writer.writeheader()
        writer.writerows(results)


def main():
    setup_logging()
    input_file = "txt2img_risky_tasks_100.csv"
    output_file = "txt2img_risky_tasks_100_output.csv"

    dataset = load_csv_to_dataset(input_file)
    results = process_dataset(dataset)
    save_results_to_csv(results, output_file)

    print(f"处理完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    main()