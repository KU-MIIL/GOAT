import argparse
import json
import os
import random

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, required=True, help='Original train data split')
parser.add_argument('--corpus_path', type=str, required=True, help='API corpus path')
parser.add_argument('--output_path', type=str, required=True, help='Preprocessed (query, positive doc) pairs')


def preprocess_retriever(args):
    train_data = []
    for filename in os.listdir(args.train_data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(args.train_data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                train_data.extend(json.load(f))

    with open(args.corpus_path, 'r', encoding='utf-8') as f:
        api_corpus = json.load(f)
    api_dict = {str(i['docid']): i['document_content'] for i in api_corpus}

    preprocessed_data = []
    for data in train_data:
        user_instruction = data['query']
        gt_documents = [api_dict[str(i['docid'])] for i in data['api_path']]
        for gtdoc in gt_documents:
            preprocessed_data.append([user_instruction, gtdoc])

    random.shuffle(preprocessed_data)

    with open(args.output_path, 'w') as f:
        json.dump(preprocessed_data, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    preprocess_retriever(args)
