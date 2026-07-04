import argparse
import json
import time

import requests
from sentence_transformers import SentenceTransformer, util

from common.llm import LLMInference
from common.util import clean_response, read_line, change_name, standardize
from prompts.CoT_it import COT_GT_SYS, COT_USER, COT_ASSIST, API_SYS, API_USER, FINAL_SYS, FINAL_USER


class DenseRetriever:
    """Retrieve candidate API documents with a (fine-tuned) SentenceTransformer."""

    def __init__(self, corpus, model_path, top_k=6):
        self.model = SentenceTransformer(model_path)
        self.documents = [item["document_content"] for item in corpus]
        self.doc_embeddings = self.model.encode(self.documents, convert_to_tensor=True)
        self.top_k = top_k

    def retrieve(self, query):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.doc_embeddings)[0]
        top = scores.topk(min(self.top_k, len(self.documents)))
        return [self.documents[i] for i in top.indices.tolist()]


class InferenceRunner:
    def __init__(self, args):
        self.args = args

        with open(args.data_path, 'r') as f:
            self.data = json.load(f)
        with open(args.corpus_path, 'r') as f:
            self.corpus = json.load(f)
        self.new_corpus = {item["docid"]: item["document_content"] for item in self.corpus}

        self.service_url = args.service_url
        self.llm_inference = LLMInference(
            model_id=args.model_path,
            token=args.hf_key,
            max_model_len=2048,
        )
        self.retriever = DenseRetriever(self.corpus, args.retriever_path)

    # ------------------------------------------------------------------ #
    # API call backends
    # ------------------------------------------------------------------ #
    def call_api(self, params, category, tool, api):
        if self.args.call_mode == "toolbench":
            return self.call_api_toolbench(params, category, tool, api)
        return self.call_api_direct(params, category, tool, api)

    def call_api_toolbench(self, params, category, tool, api):
        """Call the API through the toolbench (RapidAPI proxy) server."""
        args = self.args

        payload = {
            "category": category,
            "tool_name": standardize(tool),
            "api_name": change_name(standardize(api)),
            "tool_input": params,
            "strip": args.observ_compress_method,
            "toolbench_key": args.toolbench_key,
        }

        time.sleep(2)
        headers = {"toolbench_key": args.toolbench_key}
        try:
            response = requests.post(self.service_url, json=payload, headers=headers, timeout=30)
        except requests.exceptions.ReadTimeout:
            return json.dumps({"error": "Request timed out, please try again later.", "response": ""}), 13
        except requests.exceptions.ConnectionError:
            return json.dumps({"error": "Connection error occurred, please try again later.", "response": ""}), 13

        if response.status_code != 200:
            return json.dumps({"error": f"request invalid, data error. status_code={response.status_code}", "response": ""}), 12
        try:
            response = response.json()
        except Exception:
            return json.dumps({"error": "request invalid, data error", "response": ""}), 12

        if response["error"] == "API not working error...":
            status_code = 6
        elif response["error"] == "Unauthorized error...":
            status_code = 7
        elif response["error"] == "Unsubscribed error...":
            status_code = 8
        elif response["error"] == "Too many requests error...":
            status_code = 9
        elif response["error"] == "Rate limit per minute error...":
            print("Reach api calling limit per minute, sleeping...")
            time.sleep(10)
            status_code = 10
        elif response["error"] == "Message error..." or ("Endpoint" in response["response"] and "does not exist" in response["response"]):
            status_code = 11
        else:
            status_code = 0
        return json.dumps(response), status_code

    def call_api_direct(self, params, category, tool, api):
        """Call the real API directly instead of going through the toolbench proxy.

        Implement the request for your own API host. Must return
        (response_json_str, status_code) with the same contract as
        call_api_toolbench (status_code 0 == success).
        """
        raise NotImplementedError(
            "Direct API calling is not implemented. Fill in call_api_direct() for "
            "your API host, or run with --call_mode toolbench."
        )

    # ------------------------------------------------------------------ #
    def plan_api_sequence(self, query, domain, api_dict):
        """Retrieve candidates and let the LLM plan the API call sequence."""
        retrieved_apis = self.retriever.retrieve(query=query)

        plan_response = self.llm_inference.generate([
            {"role": "system", "content": COT_GT_SYS.format(api_list=retrieved_apis)},
            {"role": "user", "content": json.dumps(COT_USER[domain])},
            {"role": "assistant", "content": json.dumps(COT_ASSIST[domain])},
            {"role": "user", "content": query}
        ])
        plan_response = clean_response(plan_response)
        if isinstance(plan_response, dict):
            plan_response = [plan_response]

        api_seq = []
        for step in plan_response:
            api_name = step.get('api') if isinstance(step, dict) else step
            if not isinstance(api_name, str):
                continue
            docid = api_dict.get(api_name)
            if docid and docid in self.new_corpus:
                api_seq.append({docid: self.new_corpus[docid]})
        return api_seq

    def execute_api_sequence(self, query, api_seq):
        """Fill parameters and call each API in order."""
        api_call_results = []
        for api in api_seq:
            docid, document = next(iter(api.items()))
            doc = read_line(document)
            category, tool, api_name = doc['category_name'], doc['tool_name'], doc['api_name']

            prev_responses = self._compress(api_call_results)

            api_input = clean_response(self.llm_inference.generate([
                {"role": "system", "content": API_SYS},
                {"role": "user", "content": API_USER.format(query=query, document=document, api_results=prev_responses)}
            ]))

            try:
                if not isinstance(api_input, dict):
                    raise TypeError("LLM output must be a dictionary.")
                params = {change_name(key).lower(): value for key, value in api_input.items()}
                server_response, _ = self.call_api(params, category, tool, api_name)
            except Exception as e:
                server_response = json.dumps({"error": str(e)})

            api_call_results.append({
                'docid': docid,
                'llm_generation': api_input,
                'server_response': server_response
            })
        return api_call_results

    def generate_final_response(self, query, api_call_results):
        final = self.llm_inference.generate([
            {"role": "system", "content": FINAL_SYS},
            {"role": "user", "content": FINAL_USER.format(query=query, api_results=self._compress(api_call_results))}
        ])
        final_response = clean_response(final)
        if isinstance(final_response, dict):
            return final_response.get("answer", "")
        return final

    def _compress(self, api_call_results):
        limit = self.args.max_observation_length
        return [
            item['server_response'][:limit] + "...\"}"
            if len(item['server_response']) > limit else item['server_response']
            for item in api_call_results
        ]

    def run(self):
        api_dict = {read_line(item['document_content'])['api_name']: item['docid'] for item in self.corpus}

        output = []
        for d in self.data:
            query = d['query']
            domain = d['domain']  # per-item domain selects the few-shot seed
            api_seq = self.plan_api_sequence(query, domain, api_dict)
            api_call_results = self.execute_api_sequence(query, api_seq)
            final_answer = self.generate_final_response(query, api_call_results)

            output.append({
                'query': query,
                'api_path': api_call_results,
                'final_response': final_answer
            })

        with open(self.args.output_path, 'w') as f:
            json.dump(output, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the (fine-tuned) LLM')
    parser.add_argument('--retriever_path', type=str, required=True, help='Path to the fine-tuned SentenceTransformer retriever')
    parser.add_argument('--data_path', type=str, required=True, help='Evaluation dataset (json; each item has "query" and "domain")')
    parser.add_argument('--corpus_path', type=str, required=True, help='API document corpus')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--max_observation_length', type=int, default=1024, help='Max characters per API observation')
    parser.add_argument('--observ_compress_method', type=str, default="truncate", choices=["truncate", "filter", "random"])
    parser.add_argument('--call_mode', type=str, default="toolbench", choices=["toolbench", "direct"], help='how to call the APIs')
    parser.add_argument('--service_url', type=str, default="http://localhost:8080/rapidapi", help='toolbench server url (call_mode=toolbench)')
    parser.add_argument('--toolbench_key', type=str, default="", help='toolbench key to request the rapidapi service')
    parser.add_argument('--hf_key', type=str, default="", help='your huggingface token key')

    args = parser.parse_args()

    runner = InferenceRunner(args)
    runner.run()
