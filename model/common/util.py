import ast
import json
import re


def read_line(line):
    try:
        # Try parsing as JSON
        return json.loads(line)
    except json.JSONDecodeError:
        pass  # Fall back to literal_eval

    try:
        # Try parsing as Python literal
        return ast.literal_eval(line)
    except (ValueError, SyntaxError):
        pass

    raise ValueError("Input string is neither valid JSON nor a Python literal.")


def clean_response(response):
    if isinstance(response, dict):
        return response

    try:
        return json.loads(response)
    except (TypeError, ValueError):
        pass

    try:
        dict_str = re.search(r'\{[\s\S]*\}', response)
        if dict_str is None:
            return {}
        return ast.literal_eval(dict_str.group())
    except (SyntaxError, ValueError):
        return {}


def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string


def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name
