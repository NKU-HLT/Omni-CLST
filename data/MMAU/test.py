import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
def extract_answer(output_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, output_str)

    if match:
        return match.group(1)
    return output_str

def merge_model_output(json_file_path, jsonl_file_path, output_file_path):
    id_to_prediction = {}
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    id_to_prediction[item["id"]] = item.get("model_output", "")
                except json.JSONDecodeError:
                    print(f"[WARNING] jsonl：{line[:50]}...")

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item_id = item.get("id")
        if item_id in id_to_prediction:
            item["model_prediction"] = id_to_prediction[item_id]
        else:
            item["model_prediction"] = None 


    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ {output_file_path}")

@dataclass
class TestArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    model_path: Optional[str] = field(default=None, metadata={"help": "model dir"})
    out_file: Optional[str] = field(default=None, metadata={"help": "output file for test"})
    data_file: Optional[str] = field(default=None, metadata={"help": "test file"})
    force: Optional[bool] = field(default=False, metadata={"help": "force test"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "Batch size for processing"})
    think: Optional[bool] = field(default=False, metadata={"help": "whether think step by step"})
    random_drop: Optional[bool] = field(
        default=False, metadata={"help": "whether randomly drop think process for some samples"}
    )
    think_max_len: Optional[int] = field(
        default=50, metadata={"help": "Max length of think process, 0 for unlimit"}
    )
    
    def __post_init__(self):
        if self.model_path is None:
            raise ValueError("config path should not none")


def _get_message(obj_dict, think=False, think_max_len=50, random_drop=False):
    choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
    if think:
        if think_max_len <= 0:
            question_template = f"{obj_dict['question']} {choice_str} Output the thinking process in <think> </think> and final answer in <answer> </answer>."
        elif random_drop:
            question_template = f"{obj_dict['question']} {choice_str} If simple, the output is formatted as <think>\n\n</think><answer> answer here </answer>. If the question is difficult, output the thinking process (less than {think_max_len} words) in <think> </think> and final answer in <answer> </answer>."
        else:
            question_template = f"{obj_dict['question']} {choice_str} Output the thinking process(less than {think_max_len} words) in <think> </think> and final answer in <answer> </answer>."
    else:
        question_template = f"{obj_dict['question']} {choice_str}"    
    message = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": f"{obj_dict['audio_id']}"},
                {"type": "text", "text": question_template},
            ],
        }
    ]
    return message


def main():
    parser = HfArgumentParser(TestArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    qwen2_omni_processor = Qwen2_5OmniProcessor.from_pretrained(data_args.model_path)
    qwen2_omni_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        data_args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    datas = []
    datas_existing = []
    existing_ids = set()
    if data_args.out_file is not None and os.path.exists(data_args.out_file):
        with open(data_args.out_file, "r", encoding="utf8") as reader:
            for line in reader:
                line = line.strip()
                if line:  
                    datas_existing.append(json.loads(line))
    if datas_existing is not []:
        existing_ids = {d['id'] for d in datas_existing}

    with open(data_args.data_file, "r", encoding="utf8") as reader:
        datas_all = json.load(reader) 
    # Filter out existing entries
    for item in datas_all:
        if item['id'] in existing_ids:
            # print(f"Skipping existing entry with id {item['id']}")
            continue
        datas.append(item)

    all_outputs = []
    batch_size = data_args.batch_size
    output_path = data_args.out_file
    is_append = os.path.exists(output_path)
    for i in tqdm(range(0, len(datas), batch_size)):
        batch_data = datas[i : i + batch_size]

        batch_messages = []
        batch_audios = []
        for bd in batch_data:
            batch_messages.append(_get_message(bd, data_args.think, data_args.think_max_len, data_args.random_drop))
        if i == 0:
            print(f"zjh [DEBUG] Example message: {json.dumps(batch_messages[-1], indent=2, ensure_ascii=False)}")
        batch_audios, _, _ = process_mm_info(batch_messages, use_audio_in_video=False)

        text = qwen2_omni_processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)

        inputs = qwen2_omni_processor(
            text=text, audio=batch_audios, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(qwen2_omni_model.device).to(torch.bfloat16)

        generated_ids = qwen2_omni_model.generate(**inputs, max_new_tokens=256, eos_token_id=qwen2_omni_processor.tokenizer.eos_token_id)
        generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
        response = qwen2_omni_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        with open(output_path, "a" if is_append else "w", encoding="utf-8") as f:
            for input_example, model_output in zip(batch_data, response):
                original_output = model_output
                model_answer = extract_answer(original_output).strip()

                result = dict(input_example)
                result["model_output"] = model_answer
                result["model_output_ori"] = original_output
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        is_append = True
        print(f"Processed and saved batch {i//batch_size + 1}/{(len(datas) + batch_size - 1)//batch_size}")

    print(f"Results saved to {output_path}")
    merge_path = output_path.replace(".jsonl", "_submission.json")
    merge_model_output(data_args.data_file, data_args.out_file, merge_path)

if __name__ == "__main__":
    main()
