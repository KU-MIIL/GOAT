import argparse
import json
import os
import random

from transformers import AutoTokenizer

from common.util import read_line
from prompts.CoT_it import COT_GT_SYS, COT_USER, COT_ASSIST, API_SYS, API_USER, FINAL_SYS, FINAL_USER

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, required=True, help='Original train data split')
parser.add_argument('--corpus_path', type=str, required=True, help='API corpus path')
parser.add_argument('--output_path', type=str, required=True, help='Preprocessed data for finetuning')
parser.add_argument('--tokenizer_path', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help='Tokenizer path or HF id')
parser.add_argument('--data_type', type=str, required=True, choices=["base", "aug"])
parser.add_argument('--domain', type=str, required=True, choices=["financial", "food", "travel", "entertainment"])


def convert_messages_to_io(messages, tokenizer, max_input_length=2048, max_output_length=256):
    messages = messages['messages']
    last_assistant = None
    for m in reversed(messages):
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break
    if last_assistant is None:
        return None

    input_parts = []
    for m in messages:
        if m["role"] == "assistant" and m["content"] == last_assistant:
            break
        input_parts.append(f"<|{m['role']}|>\n{m['content']}")

    input_text = "\n".join(input_parts)

    if not isinstance(last_assistant, str):
        output_text = json.dumps(last_assistant, separators=(",", ":"))
    else:
        output_text = last_assistant

    output_text += tokenizer.eos_token
    full_text = input_text + output_text

    full_tokens = tokenizer(full_text, return_tensors='pt', padding=False, truncation=True, max_length=max_input_length + max_output_length)
    full_input_ids = full_tokens["input_ids"][0]

    input_tokens = tokenizer(input_text, return_tensors='pt', padding=False, truncation=True, max_length=max_input_length)
    input_len = input_tokens["input_ids"].size(1)

    labels = full_input_ids.clone()
    labels[:input_len] = -100

    return {
        "input": input_text,
        "output": output_text,
        "input_ids": full_input_ids.tolist(),
        "labels": labels.tolist()
    }


def make_masked_output(
    llm_generation: dict,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    query: str,
    document: str,
    api_results: list,
    max_input_length=2048,
    max_output_length=256,
):
    raw_output_text = json.dumps(llm_generation, separators=(",", ":"), ensure_ascii=False)
    output_text = raw_output_text + tokenizer.eos_token

    input_text = "\n".join([
        f"<|system|>\n{system_prompt}",
        f"<|user|>\n{user_prompt.format(query=query, document=document, api_results=api_results)}"
    ])
    full_text = input_text + output_text

    full_tokens = tokenizer(full_text, return_tensors="pt", truncation=True,
                            max_length=max_input_length + max_output_length)
    full_input_ids = full_tokens["input_ids"][0]

    input_len = tokenizer(input_text, return_tensors="pt").input_ids.size(1)

    labels = full_input_ids.clone()
    labels[:input_len] = -100

    # mask only the parameter *values* within the output span (token-based match)
    output_ids = full_input_ids[input_len:].tolist()
    for key, value in llm_generation.items():
        value_str = json.dumps(value, ensure_ascii=False)
        value_ids = tokenizer(value_str, add_special_tokens=False)["input_ids"]

        for i in range(len(output_ids) - len(value_ids) + 1):
            if output_ids[i:i + len(value_ids)] == value_ids:
                for j in range(len(value_ids)):
                    labels[input_len + i + j] = -100
                break  # mask the first match only

    return {
        "input": input_text,
        "output": output_text,
        "input_ids": full_input_ids.tolist(),
        "labels": labels.tolist()
    }


def preprocess_for_offline(args):
    train_data = []
    max_observation_length = 1024
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, legacy=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for filename in os.listdir(args.train_data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(args.train_data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                train_data.extend(json_data)

    with open(args.corpus_path, 'r', encoding='utf-8') as f:
        api_corpus = json.load(f)
    api_dict = {str(i['docid']): i['document_content'] for i in api_corpus}

    if args.data_type == "base":
        steps = 1
    else:
        steps = 3

    preprocessed_data = []
    for data in train_data:
        if args.data_type == "base":
            user_instructions = [data['query']]
            final_responses = [data['final_response']]
        else:
            user_instructions = [data['query'], data['paraphrase']['query1'], data['paraphrase']['query2']]
            final_responses = [data['final_response'], data['paraphrase']['response1'], data['paraphrase']['response2']]

        gt_documents = [api_dict[i['docid']] for i in data['api_path']]
        gt_apis = [i['docid'] for i in data['api_path']]

        for s in range(steps):
            user_instruction = user_instructions[s]
            final_response = final_responses[s]
            remaining_docs = {docid: doc for docid, doc in api_dict.items() if docid not in gt_apis}

            neg_documents = random.sample(
                list(remaining_docs.values()),
                min(max(0, 6 - len(gt_documents)), len(remaining_docs))
            )

            api_list = gt_documents + neg_documents
            random.shuffle(api_list)

            preprocessed_data.append(convert_messages_to_io({'messages': [
                {"role": "system", "content": COT_GT_SYS.format(api_list=api_list)},
                {"role": "user", "content": COT_USER[args.domain]},
                {"role": "assistant", "content": COT_ASSIST[args.domain]},
                {"role": "user", "content": user_instruction},
                {"role": "assistant", "content": [{"api": read_line(i)['api_name']} for i in gt_documents]}
            ]}, tokenizer))

            for i in range(len(data['api_path'])):
                prev_results = [j['server_response'][:max_observation_length] + "...\"}"
                    if len(j['server_response']) > max_observation_length else j['server_response']
                    for j in data['api_path'][:i]]

                masked = make_masked_output(
                    data['api_path'][i]['llm_generation'],
                    tokenizer,
                    system_prompt=API_SYS,
                    user_prompt=API_USER,
                    query=user_instruction,
                    document=gt_documents[i],
                    api_results=prev_results
                )

                preprocessed_data.append({
                    "input": masked["input"],
                    "output": masked["output"],
                    "input_ids": masked["input_ids"],
                    "labels": masked["labels"]
                })

            all_results = [j['server_response'][:max_observation_length] + "...\"}"
                if len(j['server_response']) > max_observation_length else j['server_response']
                for j in data['api_path']]

            preprocessed_data.append(convert_messages_to_io({'messages': [
                {"role": "system", "content": FINAL_SYS},
                {"role": "user", "content": FINAL_USER.format(query=user_instruction, api_results=all_results)},
                {"role": "assistant", "content": {"answer": final_response}}
            ]}, tokenizer))

    random.shuffle(preprocessed_data)

    with open(args.output_path, 'w') as f:
        json.dump(preprocessed_data, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    preprocess_for_offline(args)
