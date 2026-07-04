import argparse
import json
import os
import time

import pandas as pd
import requests

from common.util import clean_response, standardize, change_name
from common.llm import LLMInference
from prompts.make_apicall_param import (
    MAKE_FIRSTCALL,
    MAKE_CALL_STEP_1, STEP_1_USER, STEP_1_ASSISTANT,
    MAKE_CALL_STEP_2, STEP_2_USER, STEP_2_ASSISTANT,
)


class pipeline_runner:
    def __init__(self, args):
        self.args = args

        # load files
        self.data = self._load_json(args.tool_graph_path)

        corpus = self._load_json(args.corpus_path)
        self.corpus = {str(item.pop("docid")): item for item in corpus}

        # connectable-edge reasons (llm_pruning output): docid_x, docid_y, in_param, reason
        self.reason_corp = pd.read_csv(args.reason_path).to_dict('records')

        self.service_url = args.service_url

        self.llm_inference = LLMInference(
            model_id=args.model_id,
            token=args.hf_key,
            max_model_len=4096,
        )

    @staticmethod
    def _load_json(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    # ------------------------------------------------------------------ #
    # graph helpers
    # ------------------------------------------------------------------ #
    def build_task(self, topo):
        task = []
        for docid in topo:
            task.append({
                "docid": docid,
                "document": self.corpus[docid]['document'],
                "input_params": self.corpus[docid]['input_params'],
            })
        return task

    def find_incoming_nodes(self, subgraph, docid):
        incoming_nodes = set()
        for edge in subgraph:
            if edge["target"] == docid:
                incoming_nodes.add(edge["source"])
        return list(incoming_nodes)

    def find_reason(self, incoming_nodes, docid):
        reason = {}
        for node in incoming_nodes:
            node_reason = {}
            for item in self.reason_corp:
                # docid_x = source (out) id, docid_y = target (in) id; ids are strings in the graph
                if str(item["docid_x"]) == str(node) and str(item["docid_y"]) == str(docid):
                    node_reason[item["in_param"]] = item["reason"]
            reason[node] = node_reason
        return reason

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
        call_api_toolbench (status_code 0 == success), so the rest of the
        pipeline works unchanged.
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
    def generate_params(self, node, subgraph, api_path):
        """Generate call parameters for a single API node."""
        docid = node['docid']
        document = node['document']
        former_call = self.find_incoming_nodes(subgraph, docid)

        # root node: fill parameters from the API document only
        if not former_call:
            return self.llm_inference.generate(
                [{"role": "system", "content": MAKE_FIRSTCALL},
                 {"role": "user", "content": json.dumps({"API Document": document})}],
                num_return_sequences=1
            )

        # non-root: gather parent results, then fill parameters in two steps
        former_call_result = {}
        for n in former_call:
            call_idx = next((i for i, item in enumerate(api_path) if item["docid"] == str(n)), None)
            former_response = json.loads(api_path[call_idx]['server_response'])
            if len(former_response) > self.args.max_observation_length:
                former_response = former_response[:self.args.max_observation_length] + "...\"}"
            former_call_result[str(n)] = former_response

        reason = self.find_reason(former_call, docid)

        parameter_dict = {i: p['name'] for i, p in enumerate(node['input_params'])}
        parameters = {}
        for i, params in reason.items():
            for param_idx, r in params.items():
                parameters[param_idx] = {"docid": i, "reason": r}

        response = self.llm_inference.generate(
            [{"role": "system", "content": MAKE_CALL_STEP_1},
             {"role": "user", "content": json.dumps(STEP_1_USER)},
             {"role": "assistant", "content": json.dumps(STEP_1_ASSISTANT)},
             {"role": "user", "content": json.dumps({
                 "API Document": document,
                 "Parameter Dictionary": parameter_dict,
                 "Parameter Value": parameters,
                 "Previous Result": former_call_result,
             })}],
            num_return_sequences=1
        )

        return self.llm_inference.generate(
            [{"role": "system", "content": MAKE_CALL_STEP_2},
             {"role": "user", "content": json.dumps(STEP_2_USER)},
             {"role": "assistant", "content": json.dumps(STEP_2_ASSISTANT)},
             {"role": "user", "content": json.dumps({
                 "API Document": document,
                 "Partially Filled Parameters": clean_response(response),
             })}],
            num_return_sequences=1
        )

    def run(self):
        args = self.args
        total_iterations = len(self.data)

        for dataindex, item in enumerate(self.data):
            if dataindex < args.start:
                continue

            iteration_start_time = time.time()
            subgraph = item["subgraph"]
            topolist = item["topo_sort"]

            for topoindex, topo in enumerate(topolist):
                file_name = f"{dataindex}_{topoindex}"
                if os.path.exists(args.output_dir + file_name):
                    print(f"{file_name} already exists. Skipping...")
                    continue

                task = self.build_task(topo)
                api_path = []
                for node in task:
                    docid = node['docid']
                    document = node['document']
                    category = document['category_name']
                    tool = document['tool_name']
                    api = document['api_name']

                    response = clean_response(self.generate_params(node, subgraph, api_path))

                    param = {change_name(key).lower(): value for key, value in response.items()}
                    server_response, _ = self.call_api(param, category, tool, api)
                    api_path.append({'docid': docid, 'llm_generation': response, 'server_response': server_response})

                data = {"api_path": api_path, "subgraph": subgraph}
                with open(args.output_dir + file_name, 'w') as file:
                    json.dump(data, file, indent=4)

            iteration_duration = time.time() - iteration_start_time
            remaining_iterations = total_iterations - (dataindex + 1)
            estimated_time_remaining = remaining_iterations * iteration_duration
            hours, rem = divmod(estimated_time_remaining, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Estimated time remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tool_graph_path', type=str, required=True, help='topo-sorted subgraphs (sample_sequences output)')
    parser.add_argument('--corpus_path', type=str, required=True, help='corpus json with docid, document, input_params')
    parser.add_argument('--reason_path', type=str, required=True, help='connectable-edge csv from llm_pruning (docid_x, docid_y, in_param, reason)')
    parser.add_argument('--output_dir', type=str, required=True, help='directory to write one json per generated pipeline answer')
    parser.add_argument('--start', type=int, default=0, help='data index to resume from')
    parser.add_argument('--call_mode', type=str, default="toolbench", choices=["toolbench", "direct"], help='how to call the APIs')
    parser.add_argument('--service_url', type=str, default="http://localhost:8080/rapidapi", help='toolbench server url (call_mode=toolbench)')
    parser.add_argument('--toolbench_key', type=str, default="", help='toolbench key to request the rapidapi service (call_mode=toolbench)')
    parser.add_argument('--observ_compress_method', type=str, default="truncate", choices=["truncate", "filter", "random"], help='observation compress method')
    parser.add_argument('--max_observation_length', type=int, default=1024, help='maximum observation length')
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help='path or HF id of the LLM')
    parser.add_argument('--hf_key', type=str, default="", help='your huggingface token key')

    args = parser.parse_args()

    data_generator = pipeline_runner(args)
    data_generator.run()
