import argparse
import json
import pandas as pd
from common.llm import LLMInference
from common.util import clean_response

class pipeline_runner:
    def __init__(self, args):
        self.args = args

        with open(args.corpus_path, 'r') as f:
            corpus = json.load(f)
        self.corpus = corpus

        with open(args.edge_path, 'r') as f:
            edges = pd.read_csv(f).to_dict(orient='records')
        self.edges = json.loads(json.dumps(edges, ensure_ascii=False))

        #llama setting
        self.llm_inference = LLMInference(
            model_id=args.model_id,
            token=args.hf_key,
            max_model_len=4096,
        )

        self.sys_prompt = """
        You are an API Documentation Assistant responsible for determining if two APIs can be connected sequentially, where the output of the first API must be used as the input for the second API.
        You will be provided with:
        1. API1 Document: A dictionary containing the details of API1's output.
        2. API1 Semantic Descriptions: Natural language explanations of API1's output
        3. API2 Document: A dictionary containing the details of API2's input.
        4. API2 Semantic Descriptions: Natural language explanations of API2's input.

        Your task is to:
        1. Analyze the semantic descriptions and the provided API documents to determine if API1's output can be used as API2's input.
        2. Return True only if the information in the output of API1 can be used as a valid input for API2.
	    3. Do not return True when input of API1 can be reused in API2.
        4. Explain why the APIs are connectable or not.

        Output Format:
        - You must return a dictionary with the keys "connectable" and "reason".
        - "connectable": Return True only if API1's output can be used as API2's input, otherwise return False.
        - "reason": Provide a clear explanation describing why the APIs can or cannot be connected.

        ONLY return the dictionary as your output. DO NOT include any other words.
        """

    @staticmethod
    def _is_connectable(value):
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    def run(self):
        args = self.args
        corpus = self.corpus

        output = []
        for item in self.edges:
            out_id = item['out_id']
            in_id = item['in_id']
            doc_out = corpus[out_id]['document_content']
            doc_in = corpus[in_id]['document_content']

            llm_1 = item['output']
            llm_2 = item['input']

            prompt = f"API1 Document: {doc_out}, API1 Semantic Descriptions: {llm_1}, API2 Document: {doc_in}, API2 Semantic Descriptions: {llm_2}"
            llama_response = self.llm_inference.generate(
                [{"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": prompt}],
                num_return_sequences=1
            )

            response = clean_response(llama_response)
            # keep only edges the model judges connectable
            if not self._is_connectable(response.get('connectable')):
                continue

            output.append({
                'docid_x': out_id,
                'document_content_x': doc_out,
                'docid_y': in_id,
                'document_content_y': doc_in,
                'in_param': item['in_param'],
                'reason': response.get('reason', ''),
            })

        columns = ['docid_x', 'document_content_x', 'docid_y', 'document_content_y', 'in_param', 'reason']
        pd.DataFrame(output, columns=columns).to_csv(args.output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, required=True, help='JSON corpus with document_content, indexed by docid')
    parser.add_argument('--edge_path', type=str, required=True, help='edge csv from sentence_bert (columns: out_id, in_id, output, input, ...)')
    parser.add_argument('--output_path', type=str, required=True, help='output csv of connectable edges (used as edge_call_pruning --edge_path and apicall_generator --reason_path)')
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help='path or HF id of the LLM')
    parser.add_argument('--hf_key', type=str, default="",required=False, help='your huggingface token key')
    args = parser.parse_args()

    query_generator = pipeline_runner(args)
    query_generator.run()
    