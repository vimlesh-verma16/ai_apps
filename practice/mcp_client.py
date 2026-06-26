#!/usr/bin/env python3

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:8080/mcp"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json,text/event-stream",
}

def post_json(body, headers=None):
    payload = json.dumps(body).encode("utf-8")
    request_headers = HEADERS.copy()
    if headers:
        request_headers.update(headers)
    
    request = Request(BASE_URL, data=payload, headers=request_headers, method="POST")
    try:
        with urlopen(request) as response:
            body_text = response.read().decode("utf-8")
            return body_text, dict(response.headers)
    except HTTPError as error:
        print(f"HTTP Error {error.code}: {error.reason}")
        raise

def parse_sse(body_text):
    data_lines = [line[5:].lstrip() for line in body_text.splitlines() 
                  if line.startswith("data:")]
    return json.loads("\n".join(data_lines)) if data_lines else None

def initialize_client():
    request = {
        "jsonrpc": "2.0", "id": "1", "method": "initialize",
        "params": {"protocolVersion": "1.0", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "0.1.0"}}
    }
    body, headers = post_json(request)
    session_id = headers.get("mcp-session-id")
    return parse_sse(body), session_id

def list_tools(session_id):
    request = {
            "jsonrpc": "2.0", 
            "id": "2", 
            "method": "tools/list",
            "params": {}
        }
    body, _ = post_json(request, {"mcp-session-id": session_id})
    return parse_sse(body)

def invoke_tool(session_id, tool_name, arguments):
    request = {
        "jsonrpc": "2.0",
        "id": "3",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }
    body, _ = post_json(request, {"mcp-session-id": session_id})
    return parse_sse(body)

def main():
    # 1. Initialize


    # 3. Invoke Tool
    result = invoke_tool(session_id, "add", {"a": 10, "b": 3})
    print("Tool result:", json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
