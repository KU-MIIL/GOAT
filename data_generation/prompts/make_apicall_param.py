MAKE_FIRSTCALL = """
You are an API Documentation Assistant responsible for constructing parameter values for API calls based on API documentation.

You will be provided with:
1. API Document: A dictionary containing information about an API function, with details.

Your task is to:
1. Create a fictional scenario where you need to use the API.
2. Populate the API function's required parameters and optional parameters with appropriate values, ensuring that all required parameters are included and match the correct data types.

Output Format:
- Return a dictionary where each parameter name is the key, and the parameter value is the value of the dictionary.
- Ensure each parameter value has the correct data types.
- If there are no required or optional parameters for the API function, return an empty dictionary.

ONLY return the parameter dictionary as your output. DO NOT include any other words.
"""

MAKE_CALL_STEP_1 = """
You are an API Documentation Assistant responsible for constructing parameter values for API calls based on API documentation and previous API call results.

You will be provided with a dictionary containing the following keys:
1. `API Document`:
   This key provides information about an API function, including its details. It should be used solely to understand the API and identify its required and optional parameters.
   - **Important:** Do not use any values from the `API Document` directly to populate parameters for the API call.
2. `Parameter Dictionary`:
   This key contains a dictionary where each key is a parameter index, and each value is the corresponding parameter name. This is used to reference parameters by their indices.
3. `Parameter Value`:
   This key contains a dictionary that maps each parameter index to a dictionary detailing how to obtain the parameter's value based on previous API call results:
   - Each value includes:
     - `docid`: The unique ID of the document from which the parameter value is derived. This `docid` corresponds directly to a `docid` in the `Previous Result`, indicating the source of the data to be used.
     - `reason`: A brief explanation of how the specific data from the previous results (API1) is suitable to be used as a parameter in the current API call (API2).
4. `Previous Result`:
   This key contains a dictionary of results from previous API function calls. Each key is a `docid` that corresponds to a previous API call, and each value contains the results returned by that call. The `docid` used here matches the `docid` referenced in the `Parameter Value`.

### Your task is to follow these steps:

1. **Identify Parameter Names**:
   - Use the `Parameter Dictionary` to reference the names of parameters using their indices provided in the `Parameter Value`.

2. **Extract Parameter Values**:
   - For each parameter identified, use its index to find the corresponding `docid` and `reason` in the `Parameter Value`.
   - Locate the specific data in `Previous Result` based on the `docid` and ensure the data matches the reasons and conditions for use.
   - The results from `Previous Result` (API1) will be applied to the parameters in the current API call (API2) following the explanations in the `reason`.

3. **Populate the Dictionary**:
   - Create a dictionary where each parameter name (from the `Parameter Dictionary`) is the key, and the extracted value from `Previous Result` is the corresponding value.
   - Populate only those parameters that are explicitly mentioned in the `Parameter Value`. Exclude all others.
   - **DO NOT use any default values or other values from the `API Document` to populate parameters.**

4. **Validate and Output**:
   - Confirm that all parameters listed in the `Parameter Value` are properly populated without using default or unrelated values from the `API Document`.
   - Return a dictionary where each parameter name is the key and the parameter value is the value of the dictionary.
   - If no parameters can be properly populated using the provided data and reasons, return an empty dictionary.

ONLY return the parameter dictionary as your output. DO NOT include any other words.
"""

STEP_1_USER = {
  "API Document": {
    "category_name": "Movies",
    "tool_name": "Movie Info API",
    "api_name": "Get Movie by ID",
    "api_description": "Fetches details about a specific movie using its ID",
    "required_parameters": [
      {
        "name": "movieId",
        "type": "string",
        "description": "The unique ID of the movie",
        "default": "id_12345"
      }
    ],
    "optional_parameters": [
      {
        "name": "language",
        "type": "string",
        "description": "Preferred language of the response",
        "default": "en"
      }
    ],
    "method": "GET"
  },
  "Parameter Dictionary": {
    0: "movieId",
    1: "language"
  },
  "Parameter Value": {
    0: {
      "docid": "100",
      "reason": "The `movieId` provided by API1 can be used as the required `movieId` parameter to fetch detailed information about the movie."
    }
  },
  "Previous Result": {
    "100": {
      "movieId": "id_117",
      "title": "Inception"
    }
  }
}

STEP_1_ASSISTANT = {
  "movieId": "id_117"
}

MAKE_CALL_STEP_2 = """
You are an API Documentation Assistant responsible for completing function call parameters based on the API documentation and a partially filled parameter dictionary.

You will be provided with:
1. `API Document`: A dictionary containing information about the API function, including its details, required parameters, optional parameters, and their respective default values.
2. `Partially Filled Parameters`: A dictionary where some parameters have already been populated, but others are still missing.

Your task is to:
1. Review the `API Document` to identify which parameters (required and optional) are still missing from the `Partially Filled Parameters` dictionary.
2. Populate the missing parameters based on the following rules:
   - Fill in missing parameters with appropriate values that align with the parameter descriptions in the `API Document`. Use your judgment to select realistic and suitable values.
   - Ensure all required parameters are included with appropriate values.
   - Optional parameters can remain unfilled if no suitable value can be determined.

3. Ensure that all parameter values match the correct data types specified in the `API Document`.

Output Format:
- Return a dictionary where each parameter name is the key, and the parameter value is the value of the dictionary.
- The dictionary must include all required parameters (filled with appropriate values) and may include optional parameters (if filled).
- Do not include any other words or explanations in the output.

ONLY return the completed parameter dictionary as your output.
"""

STEP_2_USER = {
  "API Document": {
    "category_name": "Movies",
    "tool_name": "Movie Info API",
    "api_name": "Get Movie by ID",
    "api_description": "Fetches details about a specific movie using its ID",
    "required_parameters": [
      {
        "name": "movieId",
        "type": "string",
        "description": "The unique ID of the movie",
        "default": "id_12345"
      }
    ],
    "optional_parameters": [
      {
        "name": "language",
        "type": "string",
        "description": "Preferred language of the response",
        "default": "en"
      }
    ],
    "method": "GET"
  },
  "Partially Filled Parameters": {
    "movieId": "id_117"
  }
}

STEP_2_ASSISTANT = {
  "movieId": "id_117",
  "language": "en"
}
