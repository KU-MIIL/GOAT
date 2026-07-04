import argparse
import json

from common.util import clean_response
from common.llm import LLMInference


class pipeline_runner:
    def __init__(self, args):
        self.args = args

        with open(args.data_path, 'r') as f:
            self.data = json.load(f)

        self.llm_inference = LLMInference(
            model_id=args.model_id,
            token=args.hf_key,
            max_model_len=4096,
        )

        self.sys_prompt = """
        You are a query validation assistant responsible for evaluating the completeness and relevance of user queries and responses.

        You will receive a dictionary containing the following keys:
        1. 'user_query': A natural language question or request from the user.
        2. 'subinstructions': A list of sub-tasks or steps that should be performed to fulfill the user query.
        3. 'response': The generated response to the user query.

        ### Your task is to follow these steps:
        1. **Assess User Query Completeness:**
        - Analyze the 'user_query' to determine if it contains enough detail to logically derive the subinstructions.
        - Compare the 'user_query' with the provided 'subinstructions.'
        - Identify any gaps or ambiguities in the 'user_query' that would prevent the accurate formulation of the subinstructions.
        - If all subinstructions can be directly inferred from the 'user_query' without assumption or missing context, consider the query complete.

        2. **Evaluate Response Appropriateness:**
        - Examine the provided 'response' in relation to the 'user_query' and 'subinstructions.'
        - Determine if the response directly addresses the 'user_query' and aligns with the subinstructions.
        - Ensure the response is accurate, relevant, and logically derived from the subinstructions.
        - Identify if the response omits critical information or introduces irrelevant content.

        ### Output Format:
        - Return a dictionary with the following keys:
            - "result": Boolean value (True or False). Return True if both the user query is complete and the response is appropriate; otherwise, return False.
            - "thought": A brief explanation of why the result is True or False. Summarize the analysis of query completeness and response appropriateness.

        ### Important Notes:
        - If the user query lacks necessary detail to derive the subinstructions, the result must be False, regardless of the quality of the response.
        - If the response does not adequately address the user query, the result must also be False, even if the query is complete.
        - Focus on clarity and accuracy in your thought process. Provide concise but thorough reasoning for the returned result.

        ONLY return the output dictionary. DO NOT include any other words or commentary.
        """

    @staticmethod
    def _is_valid(value):
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    def run(self):
        output = []
        for data in self.data:
            prompt = {
                "user query": data["query"],
                "subinstructions": data["subinstructions"],
                "response": data["final_response"]
            }

            llama_response = clean_response(self.llm_inference.generate(
                [{"role": "system", "content": self.sys_prompt},
                 {"role": "user", "content": json.dumps(prompt)}],
                num_return_sequences=1
            ))

            # keep only validated instances; emit the final release schema
            if not self._is_valid(llama_response.get("result")):
                continue

            output.append({
                "query": data["query"],
                "subinstructions": data["subinstructions"],
                "api_path": data["api_path"],
                "final_response": data["final_response"],
                "domain": self.args.domain,
            })

        with open(self.args.output_path, 'w') as file:
            json.dump(output, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='query/response output (from query_response_generator)')
    parser.add_argument('--output_path', type=str, required=True, help='final validated dataset (query, subinstructions, api_path, final_response, domain)')
    parser.add_argument('--domain', type=str, required=True, choices=["financial", "food", "travel", "entertainment"], help='domain stamped on every emitted item')
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help='path or HF id of the LLM')
    parser.add_argument('--hf_key', type=str, default="", help='your huggingface token key')

    args = parser.parse_args()

    runner = pipeline_runner(args)
    runner.run()
