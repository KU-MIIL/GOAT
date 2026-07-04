import argparse
import json
import time

import networkx as nx
import pandas as pd
import requests

from common.util import clean_response, standardize, change_name
from common.llm import LLMInference
from prompts.make_apicall import MAKE_FIRSTCALL, MAKE_CALL


# system prompt for the connectivity-check stage
PRUNE_SYS_PROMPT = """
You are an API Documentation Assistant responsible for determining if the information from the result of the first API call is used in the parameters of the second API call.
You will be provided with:
1. api_result: A result from the first API call.
2. llm_result: Parameters and their values for calling next API.

Your task is to:
1. Analyze the contents of api_result to determine if it was used as input in llm_result.
2. Provide an explanation about whether or not the first API result influenced the parameters of the next API call.

Output Format:
- You must return a dictionary with the keys "connectable" and "reason".
- "connectable": Return True if api_result was used in llm_result, otherwise return False.
- "reason": Provide a clear explanation describing why api_result was or was not used as part of llm_result.

ONLY return the dictionary as your output. DO NOT include any other words.
"""

# server responses that indicate a failed / invalid API call
ERROR_PATTERNS = [
    "Authentication failed, probably because of invalid/missing API key.",
    "Access is denied. Retrying will not help.",
    "API doesn't exists",
    "\\nexpression cannot contain assignment",
    "The deployment could not be found on Vercel.",
    "The API is unreachable, please contact the API provider",
]


class pipeline_runner:
    def __init__(self, args):
        self.args = args

        # candidate edges from the previous step (pruned edges csv)
        df = pd.read_csv(args.edge_path)
        self.edges = json.loads(df.to_json(orient='records', force_ascii=False))

        self.service_url = args.service_url

        self.llm_inference = LLMInference(
            model_id=args.model_id,
            token=args.hf_key,
            max_model_len=4096,
        )

    # ------------------------------------------------------------------ #
    # API call backends
    # ------------------------------------------------------------------ #
    def call_api(self, params, category, tool, api):
        """Dispatch to the configured calling backend."""
        if self.args.call_mode == "toolbench":
            return self.call_api_toolbench(params, category, tool, api)
        return self.call_api_direct(params, category, tool, api)

    def call_api_toolbench(self, params, category, tool, api):
        """Call the API through the toolbench (RapidAPI proxy) server.

        Returns (response_json_str, status_code); status_code 0 == success.
        """
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
        call_api_toolbench (status_code 0 == success), so that the rest of the
        pipeline (edge_calling / build_graph) works unchanged.
        """
        # TODO: build the endpoint / headers / payload for your API host from
        #       (category, tool, api) and the LLM-generated `params`. Skeleton:
        #
        # url = f"{self.service_url}/{standardize(tool)}/{change_name(standardize(api))}"
        # try:
        #     resp = requests.post(url, json=params, timeout=30)
        # except requests.exceptions.RequestException as e:
        #     return json.dumps({"error": str(e), "response": ""}), 13
        # if resp.status_code != 200:
        #     return json.dumps({"error": f"status_code={resp.status_code}", "response": ""}), 12
        # return json.dumps(resp.json()), 0
        raise NotImplementedError(
            "Direct API calling is not implemented. Fill in call_api_direct() for "
            "your API host, or run with --call_mode toolbench."
        )

    # ------------------------------------------------------------------ #
    # Stage 1: realize each candidate edge by actually calling both APIs
    # ------------------------------------------------------------------ #
    def edge_calling(self):
        edge_calls = []
        for item in self.edges:
            doc_out = item['document_content_x']
            id_out = item['docid_x']
            doc_in = item['document_content_y']
            id_in = item['docid_y']
            in_param = item['in_param']
            reason = item['reason']

            llm_result = {}
            api_result = {}

            # first api
            category = json.loads(doc_out)['category_name']
            tool = json.loads(doc_out)['tool_name']
            api = json.loads(doc_out)['api_name']

            response = self.llm_inference.generate(
                [{"role": "system", "content": MAKE_FIRSTCALL},
                 {"role": "user", "content": "API Document: " + doc_out}],
                num_return_sequences=1
            )
            response = clean_response(response)
            llm_result[id_out] = response
            params = {change_name(key).lower(): value for key, value in response.items()}

            server_response, status_code = self.call_api(params, category, tool, api)
            api_result[id_out] = server_response

            # second api (uses the first api's result)
            category = json.loads(doc_in)['category_name']
            tool = json.loads(doc_in)['tool_name']
            api = json.loads(doc_in)['api_name']

            prompt = "API Document: " + doc_in + "API Call Results: " + server_response + "Reason: " + reason
            response = self.llm_inference.generate(
                [{"role": "system", "content": MAKE_CALL},
                 {"role": "user", "content": prompt}],
                num_return_sequences=1
            )
            response = clean_response(response)
            llm_result[id_in] = response
            params = {change_name(key).lower(): value for key, value in response.items()}

            server_response, status_code = self.call_api(params, category, tool, api)
            api_result[id_in] = server_response

            edge_calls.append({"llm_result": llm_result, "api_result": api_result, "in_param": in_param})

        return edge_calls

    # ------------------------------------------------------------------ #
    # Stage 2: keep only edges where the first result was actually used
    # ------------------------------------------------------------------ #
    def prune_edges(self, edge_calls):
        decisions = []
        for item in edge_calls:
            llm_result = list(item['llm_result'].values())[1]
            api_result = list(item['api_result'].values())[0]

            llama_response = self.llm_inference.generate(
                [{"role": "system", "content": PRUNE_SYS_PROMPT},
                 {"role": "user", "content": f"api_result: {api_result}, llm_result: {llm_result}"}],
                num_return_sequences=1
            )
            decisions.append(clean_response(llama_response))
        return decisions

    # ------------------------------------------------------------------ #
    # Stage 3: drop failed calls and build the final weighted graph
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_connectable(value):
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    def build_graph(self, edge_calls, decisions):
        # weighted directed graph; edge weight is the target's input-parameter
        # index (in_param), so sample_sequences can read it back as an edgelist.
        graph = nx.MultiDiGraph()
        for idx, decision in enumerate(decisions):
            if not self._is_connectable(decision.get('connectable')):
                continue

            api_result = edge_calls[idx]['api_result']
            api1 = list(api_result.values())[0]
            api2 = list(api_result.values())[1]

            if any((pattern in api1) or (pattern in api2) for pattern in ERROR_PATTERNS):
                continue

            out_node = list(api_result.keys())[0]
            in_node = list(api_result.keys())[1]
            graph.add_edge(str(out_node), str(in_node), weight=edge_calls[idx]['in_param'])

        return graph

    # ------------------------------------------------------------------ #
    def run(self):
        edge_calls = self.edge_calling()
        decisions = self.prune_edges(edge_calls)

        graph = self.build_graph(edge_calls, decisions)
        nx.write_weighted_edgelist(graph, self.args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge_path', type=str, required=True, help='pruned edge csv from llm_pruning')
    parser.add_argument('--output_path', type=str, required=True, help='output weighted edgelist of the final graph (consumed by sample_sequences --graph_path)')
    parser.add_argument('--call_mode', type=str, default="toolbench", choices=["toolbench", "direct"], help='how to call the APIs')
    parser.add_argument('--service_url', type=str, default="http://localhost:8080/rapidapi", help='toolbench server url (call_mode=toolbench)')
    parser.add_argument('--toolbench_key', type=str, default="", help='toolbench key to request the rapidapi service (call_mode=toolbench)')
    parser.add_argument('--observ_compress_method', type=str, default="truncate", choices=["truncate", "filter", "random"], help='observation compress method')
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help='path or HF id of the LLM')
    parser.add_argument('--hf_key', type=str, default="", help='your huggingface token key')

    args = parser.parse_args()

    data_generator = pipeline_runner(args)
    data_generator.run()
