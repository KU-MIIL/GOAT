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

        # ---- query synthesis prompt (+ one-shot) ----
        self.query_sys_prompt = """
        You are an API Documentation Assistant. Your role is to interpret a list of subinstructions—understanding that each subinstruction is planned, executed, and possibly leads to creating or adapting subsequent subinstructions based on prior outcomes—and convert them into a single, high-level user query that reflects their collective intent without revealing any internal steps or technical API jargon.

        ### Provided Information:
        - Subinstructions: A sequence of iterative steps working toward a single overarching objective. They are planned and executed in order, and each result can influence the creation or modification of the next subinstruction.

        ### Your Task:
        1. Infer the broader purpose by analyzing how these subinstructions connect logically and build upon each other’s results.
        2. Synthesize them into one natural, user-friendly query that preserves crucial details and dependencies but does not mention the subinstructions themselves.
        3. Represent information at a high level wherever possible, but retain all specific details (e.g., IDs, names, dates) from the **first subinstruction** exactly as they are.
        4. For subinstructions after the first one, prioritize connecting them through context (e.g., "first video," "latest episode") rather than using specific identifiers unless absolutely necessary.
        5. Ensure that every subinstruction meaningfully contributes to the final query, preventing any extraneous or unaligned steps.
        6. Avoid any technical language or references to specific APIs in the final query.

        ### Guidelines:
        - Include all essential identifiers or conditions (e.g., names, dates, relevant context) from the subinstructions. Do not omit or generalize key details from the **first subinstruction**.
        - For subsequent subinstructions, derive necessary information from the results of prior steps whenever possible. Use contextual references to maintain continuity without repeating specific identifiers unless required.
        - Reflect the necessary sequence or dependency implied by the iterative nature of subinstructions without explicitly describing each step.
        - The final query must reflect the intent of **all subinstructions** to ensure no step becomes irrelevant or disconnected.

        ### Output Format:
        - Return only a dictionary with two keys:
        - "thought": A short explanation of how you derived the final query from the subinstructions.
        - "query": The single-sentence user query representing the overall goal.

        DO NOT include API names or technical jargon from the API.
        ONLY return the dictionary as your output. DO NOT include any other words.
                """

        self.query_user_prompt = """
        Subinstructions: [Get all user records from the database. Retreive profile information for Alice, most recently active user.]
        """

        self.query_assist_prompt = """
        {
            "thought": "The subinstructions involve retrieving all user records from the database and identifying the profile of the most recently active user. The overarching goal is to get the profile information of the latest active user.",
            "query": "Retrieve the profile information of the most recently active user from the database."
        }
        """

        # ---- final answer prompt ----
        self.response_sys_prompt = """
        You are an answer generation assistant tasked with providing natural language responses to user queries by analyzing API call results.

        You will be provided a dictionary containing the following keys:
        1. 'User Query': A natural language question or request from the user.
        2. 'API Call Result': A list of dictionaries, each representing a step or subinstruction carried out to fulfill the user query. Each dictionary contains:
        - 'subinstruction': A brief description of the step taken.
        - 'api response': The actual data or result obtained from executing the subinstruction.

        ### Your task is to follow these steps:

        1. ** Analyze API Call Result: **
        - Examine each dictionary in the 'API Call Result' list.
        - Understand the purpose of each 'subinstruction' and the corresponding 'api response.'
        - Identify how each 'api response' contributes to answering the 'User Query.'
        - If necessary, combine results from multiple subinstructions to generate a comprehensive answer.

        2. ** Generate Final Answer: **
        - Construct a coherent and natural response to the 'User Query' based on the collected information from 'API Call Result.'
        - Use clear and concise language, phrasing the answer in a way that feels conversational and human-like.
        - Ensure the final response directly addresses the user's request without unnecessary detail.
        - Summarize or filter the results if needed, prioritizing the most relevant information.
        - If any subinstruction does not yield meaningful data, exclude it from the final answer and focus on the most relevant results.

        Output Format:
        - Return a dictionary with the following keys:
        - "thought": Provide a concise summary of how the API Call Result was analyzed, how relevant subinstructions were chosen, and how they were combined to address the User Query.
        - "final_answer": A natural language response to the User Query, synthesized from the API Call Result. This should sound as if an agent is directly responding to the user.

        DO NOT include API jargon or technical terms in the final answer. Only present information in user-friendly, natural language.
        Focus on delivering the information as if you are the final point of communication with the user.
        ONLY return the output dictionary. DO NOT include any other words.
        """

    def generate_query(self, data):
        """Synthesize all sub-instructions into a single high-level user query."""
        subinstructions = [node["sub_instruction"] for node in data["api_path"]]
        data['subinstructions'] = subinstructions

        llama_response = self.llm_inference.generate(
            [{"role": "system", "content": self.query_sys_prompt},
             {"role": "user", "content": self.query_user_prompt},
             {"role": "assistant", "content": self.query_assist_prompt},
             {"role": "user", "content": "Subinstructions: " + str(subinstructions)}],
            num_return_sequences=1
        )
        response = clean_response(llama_response)
        try:
            data["query"] = response["query"]
        except (SyntaxError, ValueError, KeyError):
            data["query"] = ""

    def generate_final_response(self, data):
        """Generate the natural-language final answer to the synthesized query."""
        result = []
        for node in data["api_path"]:
            response = node["server_response"]
            if len(response) > 1024:
                response = response[:1024] + "...\"}"
            result.append({"subinstruction": node["sub_instruction"], "api response": response})

        prompt = {
            "User Query": data["query"],
            "API Call Result": result
        }

        llama_response = self.llm_inference.generate(
            [{"role": "system", "content": self.response_sys_prompt},
             {"role": "user", "content": json.dumps(prompt)}],
            num_return_sequences=1
        )
        llama_response = clean_response(llama_response)
        try:
            data['final_response'] = llama_response["final_answer"]
        except (SyntaxError, ValueError, KeyError):
            data['final_response'] = ""

    def run(self):
        for data in self.data:
            self.generate_query(data)
            self.generate_final_response(data)

        with open(self.args.output_path, 'w') as file:
            json.dump(self.data, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='subinstruction output (from subinst_generator)')
    parser.add_argument('--output_path', type=str, required=True, help='output json with query, subinstructions, and final_response added')
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help='path or HF id of the LLM')
    parser.add_argument('--hf_key', type=str, default="", help='your huggingface token key')

    args = parser.parse_args()

    runner = pipeline_runner(args)
    runner.run()
