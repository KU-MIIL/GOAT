import argparse
import json
from common.llm import LLMInference
from common.util import clean_response
from prompts.build_semantic import BUILD_SEMANTIC

class pipeline_runner:
    def __init__(self, args):
        self.args = args

        with open(args.corpus_path, 'r') as f:
            corpus = json.load(f)
        self.corpus = corpus

        #set up for llama inference
        self.llm_inference = LLMInference(
            model_id=args.model_id,
            token=args.hf_key,
            max_model_len=4096,
        )

    def run(self):
        output = []
        for api in self.corpus:
            document = api['document_content']

            llama_response = self.llm_inference.generate(
                [{"role": "system", "content": BUILD_SEMANTIC},
                {"role": "user", "content": "API Document: " + str(document)}],
                num_return_sequences=1
            )

            response = clean_response(llama_response) # parse dictionary
            output.append(response)
        
        with open(self.args.output_path, 'w') as f:
            json.dump(output, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, required=True, help='corpus_path')
    parser.add_argument('--output_path', type=str,required=True, help='output path')
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help='path or HF id of the LLM')
    parser.add_argument('--hf_key', type=str,required=True, help='your huggingface token key')

    args = parser.parse_args()

    query_generator = pipeline_runner(args)
    query_generator.run()