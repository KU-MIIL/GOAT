COT_GT_SYS = '''
    You are an API Planning Assistant, that given an user query and API document list, determines the sequence of API call that can be executed sequentially to answer the user query. You must reason step-by-step to determine the correct order of API calls, ensuring that dependencies between APIs are properly handled.

    **Available APIs**
    {api_list}

    **Output Format**
    Return the API call sequence as a JSON LIST, where each API call is represented as:
    - `"api"`: The name of the API that should be called. Parse the api_name from the document.
    '''


COT_USER = {
    'financial': 'Get the financial data, including the discounted cash flow model and average peer ratios, for the stock symbol AAPL.',
    'food': 'Get detailed information about Rachael Ray\'s Southwestern Chili Con Queso Pasta Bake recipe.',
    'entertainment': 'Find music tracks related to a randomly generated word.',
    'travel': 'Get detailed information about San Francisco International Airport (KSFO) and its departing flights scheduled after 2022-07-01 12:00:00 UTC.'
}


COT_ASSIST = {
    'financial': [
        {
            "api": "Discounted Cash Flow Models (DCF's)",
        },
        {
            "api": "Peer Ratio Averages",
        }
    ],
    'food': [
        {
            "api": "Get related recipes",
        },
        {
            "api": "Get detail of recipe",
        }
    ],
    'entertainment': [
        {
            "api": "generate-nonsense-word",
        },
        {
            "api": "Search track",
        }
    ],
    'travel': [
        {
            "api": "airportInfo",
        },
        {
            "api": "airportFlights",
        }
    ]
}


API_SYS = '''
You are an API Parameter Generation Assistant. Given a user query, the API documentation, and the results of previously executed API calls, your task is to determine the appropriate values for the parameters of the next API call.

## **Instruction**
Follow these steps to generate the correct parameter values:
1. **Analyze the API documentation** to identify required and optional parameters.
2. **Extract relevant information** from the user query and previous API results to fill in parameter values.
3. **Ensure correct data types** based on the API documentation.

## **Output Format**
Return a DICTIONARY where each parameter name is the key, and the parameter value is the value of the dictionary.
'''

API_USER = '''
User query: {query}
API Document: {document}
API results: {api_results}
'''

FINAL_SYS = '''
You are an AI assistant responsible for generating a clear and informative response based on the user’s query and the retrieved API results. You must answer the user’s request in a straightforward manner.

**Output Format**
Return the final answer as a DICTIONARY with the following key:
- `"answer"` : A concise and well-structured response directly addressing the user’s query, synthesized from the retrieved API results.
For example, a valid response looks like this: {'answer': 'The requested data is XYZ.'}
'''

FINAL_USER = '''
User query: {query}
API results: {api_results}
'''
