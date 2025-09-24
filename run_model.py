# LVLM-LP/run_model.py
import os
import argparse

import cv2
import json
import numpy as np
from tqdm import tqdm

from model import build_model
from dataset import build_dataset
from utils.func import get_chunk
from utils.prompt import Prompter

import torch

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU only")
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


def load_image(img_path):
    """Load image from disk and convert to RGB."""
    image = cv2.imread(img_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_model_with_prompt(model, image, prompt):
    """Run model forward pass and return outputs."""
    return model.forward_with_probs(image, prompt)


def run_gpt4v_model(model, img_path, prompt):
    """Run GPT4V forward pass (image path list + text)."""
    image = [img_path]  # GPT4V expects paths, not RGB arrays
    return model.forward(image, prompt)


def build_gpt4v_output(ins, img_id, response, extra_keys):
    """Construct GPT4V output dictionary."""
    out = {
        "image": img_id,
        "question": ins['question'],
        "label": ins["label"],
        "response": response,
    }
    for key in extra_keys:
        out[key] = ins[key]
    return out


def build_output(ins, img_id, args, response, output_ids, logits, extra_keys):
    """Construct a dictionary of results for one sample."""
    if len(logits) <= args.token_id:
        return None

    out = {
        "image": img_id,
        "model_name": args.model_name,
        "question": ins['question'],
        "label": ins["label"],
        "response": response,
        "output_ids": output_ids.tolist(),
        "logits": logits.tolist()[args.token_id],
        # "probs": probs.tolist()[args.token_id],  # optional
    }
    for key in extra_keys:
        out[key] = ins[key]
    return out


def process_gpt4v_instance(ins, model, extra_keys, ans_file):
    """Process a single dataset instance with GPT4V and write result to file."""
    img_id = ins['img_path'].split("/")[-1]
    response = run_gpt4v_model(model, ins['img_path'], ins['question'])
    print(response)

    out = build_gpt4v_output(ins, img_id, response, extra_keys)
    ans_file.write(json.dumps(out) + "\n")


def process_instance(ins, model, args, extra_keys, ans_file):
    """Process a single dataset instance and write result to file."""
    img_id = ins['img_path'].split("/")[-1]
    image = load_image(ins['img_path'])
    response, output_ids, logits, probs = run_model_with_prompt(model, image, ins['question'])

    out = build_output(ins, img_id, args, response, output_ids, logits, extra_keys)
    if out:  # only write if logits are valid
        ans_file.write(json.dumps(out) + "\n")


def get_model_output(args, data, model, extra_keys, answers_file):
    ans_file = open(answers_file, 'w')

    for ins in tqdm(data):
        if args.model_name == "GPT4V":
            process_gpt4v_instance(ins, model, extra_keys, ans_file)
        else:
            process_instance(ins, model, args, extra_keys, ans_file)

        ans_file.flush()

    ans_file.close()


def main(args):
    model = build_model(args)

    prompter = Prompter(args.prompt, args.theme)

    data, extra_keys = build_dataset(args.dataset, args.split, prompter)
    if args.num_samples is not None:
        if args.sampling == 'first':
            data = data[:args.num_samples]
        elif args.sampling == "random":
            np.random.shuffle(data)
            data = data[:args.num_samples]
        else:
            labels = np.array([ins['label'] for ins in data])
            classes = np.unique(labels)
            data = np.array(data)
            final_data = []
            for cls in classes:
                cls_data = data[labels == cls]
                idx = np.random.choice(range(len(cls_data)), args.num_samples, replace=False)
                final_data.append(cls_data[idx])
            data = list(np.concatenate(final_data))

    data = get_chunk(data, args.num_chunks, args.chunk_idx)

    if not os.path.exists(f"./output/{args.model_name}/"):
        os.makedirs(f"./output/{args.model_name}/")

    get_model_output(args, data, model, extra_keys, args.answers_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')
    parser.add_argument("--model_name", default="LLaVA-7B")
    parser.add_argument("--model_path", default="extern/liuhaotian/llava-v1.5-7b")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--sampling", choices=['first', 'random', 'class'], default='first')
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset", default="MathVista")
    parser.add_argument("--prompt", default='oeh')
    parser.add_argument("--theme", default='unanswerable')
    parser.add_argument("--answers_file", type=str, default="./output/LLaVA-7B/LLaVA_VizWiz_val_oeh.jsonl")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--token_id", type=int, default=0)

    main(parser.parse_args())
