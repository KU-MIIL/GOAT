import argparse
import json

from common.util import clean_response
from common.llm import LLMInference


class pipeline_runner:
    def __init__(self, args):
        self.args = args

        with open(args.data_path, 'r') as f:
            self.data = json.load(f)

        with open(args.corpus_path, 'r') as f:
            self.corpus = json.load(f)

        self.llm_inference = LLMInference(
            model_id=args.model_id,
            token=args.hf_key,
            max_model_len=4096,
        )

        # one-shot example
        self.one_shot_user = {
            "API Document": "{\"category_name\":\"User Information\",\"tool_name\":\"User Lookup\",\"api_name\":\"/get_user_details\",\"api_description\":\"Retrieves detailed information about a user based on their unique ID.\",\"required_parameters\":[{\"name\":\"userId\",\"type\":\"integer\",\"description\":\"The unique identifier for the user whose details need to be retrieved.\",\"default\":null}],\"optional_parameters\":[],\"method\":\"GET\"}",
            "API call": {"userId": 12345},
            "Previous API Response": {"users": [{"userId": 12345, "name": "Alice", "age": 30, "isActive": True, "lastLogin": "2024-11-18T10:00:00Z"}, {"userId": 67890, "name": "Bob", "age": 25, "isActive": False, "lastLogin": "2024-10-25T14:30:00Z"}]}
        }

        self.one_shot_assist = {
            "thought": "The 'userId' 12345 was selected from the previous response because it corresponds to an active user with a recent login.",
            "plan": "Retrieve detailed profile information form the most recently active user."
        }

        self.sys_prompt = """
        You are an instruction generation assistant for generating language instruction that enables execution of given API call.

        You will be provided a dictionary containing the following keys:
        1. 'API Document': A structured description of the API, including its purpose, required and optional parameters, and any relevant context about its functionality.
        2. 'API call': A dictionary of specific parameter values intended for execution of the API call. You must generate language instruction that enables execution of this call.
        3. 'Previous API Response': The output or result from preceding API calls. Some values in 'API call' references values in this result. If this is empty, it should not be referenced.

        ### Your task is to follow these steps:

        1. **Classify Parameters in 'API call':
            - For each key in 'API call', check if its value can be directly derived from the 'Previous API Response'.
            - Classify keys into two groups:
                a. Derived Parameters: Parameters whose values are obtained from the 'Previous API Response'.
                b. Fixed Parameters: Parameters with values that are not contained in 'Previous API Response'.

        2. **Generate Language Instruction:
            - Generate a clear and concise language instruction that enables the execution of the 'API call'.
            - Use the 'API Document' to understand the intent of the 'API call' and ensure that the generated instruction aligns with its goal. The instruction must be goal-oriented, actionable, and contextually accurate.
            - Incorporate the parameter classification:
                a. Derived Parameters: For parameters classified as derived, include in the instruction a detailed explanation of how their values are obtained from the 'Previous API Response.' Clearly reference the specific part or context of the 'Previous API Response' used to derive these values.
                    Do not include the derived value itself in the instruction. Instead, describe the reasoning behind its selection, such as it being the first item, the most recent value, the largest attribute, or another logical criterion. The reasoning must be explicit and actionable.
                b. Fixed Parameters: For parameters classified as fixed, include their specific values directly in the instruction. Ensure these values are explicitly stated to avoid ambiguity.
            - The instruction should naturally integrate both types of parameters and describe the action to be performed in a clear and executable manner.

        Output Format:
        - Return a dictionary with the following keys:
            - "thought": Provide a brief but clear explanation of your reasoning process, including how parameters were classified, how derived values were selected, and how they were incorporated into the instruction.
            - "instruction": Generate a concise, goal-oriented sentence that describes the action required to execute the given API call. Ensure the instruction integrates both derived and fixed parameters appropriately, specifying derived parameter contexts, the reasoning for their selection, and explicitly stating fixed parameter values.

        DO NOT use the vague terms such as "use the obtained value" or "from specific values."
        DO NOT include parameter names or technical jargon from the API. Translate these into natural language descriptions of their role or value.
        ONLY return the output dictionary. DO NOT include any other words.
        """

    def find_incoming_nodes(self, subgraph, docid):
        incoming_nodes = set()
        for edge in subgraph:
            if edge["target"] == docid:
                incoming_nodes.add(edge["source"])
        return list(incoming_nodes)

    def run(self):
        args = self.args

        final_data = []
        for data in self.data:
            for item in data['api_path']:
                document = None
                for doc in self.corpus:
                    if doc['docid'] == int(item['docid']):
                        document = doc['document_content']
                        break

                formal_nodes = self.find_incoming_nodes(data['subgraph'], item['docid'])
                formal_response = []
                for n in formal_nodes:
                    matching_indices = [j for j, it in enumerate(data['api_path']) if it["docid"] == str(n)]
                    selected_index = max(matching_indices)
                    formal_response.append(json.loads(data['api_path'][selected_index]['server_response']))

                prompt = {
                    "API Document": document,
                    "Parameter values for API call": item['llm_generation'],
                    "Previous API Response": formal_response
                }

                llama_response = self.llm_inference.generate(
                    [{"role": "system", "content": self.sys_prompt},
                     {"role": "user", "content": json.dumps(self.one_shot_user)},
                     {"role": "assistant", "content": json.dumps(self.one_shot_assist)},
                     {"role": "user", "content": json.dumps(prompt)}],
                    num_return_sequences=1
                )
                response = clean_response(llama_response)
                try:
                    response = response["instruction"]
                except (SyntaxError, ValueError, KeyError):
                    response = ""

                item['sub_instruction'] = response
            final_data.append(data)

        with open(args.output_path, 'w') as file:
            json.dump(final_data, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, required=True, help='corpus json with docid, document_content')
    parser.add_argument('--data_path', type=str, required=True, help='pipeline api-call output (from pipeline_apicall)')
    parser.add_argument('--output_path', type=str, required=True, help='output json with a sub_instruction added to each step')
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help='path or HF id of the LLM')
    parser.add_argument('--hf_key', type=str, default="", help='your huggingface token key')

    args = parser.parse_args()

    runner = pipeline_runner(args)
    runner.run()
