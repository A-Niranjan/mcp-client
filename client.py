import asyncio
import os
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
import json

# MCP Imports
from mcp import ClientSession, StdioServerParameters, Tool as McpTool # Renamed McpTool to avoid name clash
from mcp.client.stdio import stdio_client

# Gemini Imports
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, Tool as GeminiTool, FunctionDeclaration

# Env variable loader
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

# --- Gemini Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file.")
genai.configure(api_key=GOOGLE_API_KEY)
# --- End Gemini Configuration ---

def convert_mcp_tool_to_gemini(mcp_tool: McpTool) -> FunctionDeclaration:
    """Converts MCP Tool schema to Gemini FunctionDeclaration."""

    # 1. Prepare the parameters dictionary completely beforehand
    gemini_params = {
        "type": "object",
        "properties": mcp_tool.inputSchema.get("properties", {}),
        # Add other potential top-level schema keys here if needed (e.g., 'description')
    }

    # 2. Conditionally add 'required' key to the dictionary
    # Gemini expects 'required' to be a list of strings.
    if "required" in mcp_tool.inputSchema:
        required_fields = mcp_tool.inputSchema["required"]
        # Ensure it's a list and not empty before adding
        if isinstance(required_fields, list) and required_fields:
             gemini_params["required"] = required_fields
        elif isinstance(required_fields, list) and not required_fields:
             # If required is an empty list, don't add the key,
             # as an empty 'required' list might not be valid in OpenAPI schema.
             pass
        else:
            # Handle case where 'required' might be something else (though schema usually dictates list)
            print(f"Warning: 'required' field in schema for tool '{mcp_tool.name}' is not a list, skipping.")


    # 3. Create FunctionDeclaration with the fully constructed parameters dictionary
    func_decl = FunctionDeclaration(
        name=mcp_tool.name,
        description=mcp_tool.description or f"Tool named {mcp_tool.name}", # Gemini requires a description
        parameters=gemini_params # Pass the prepared dictionary
    )

    return func_decl

class MCPClient:
    def __init__(self):
        """Initializes the MCP Client."""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # --- Gemini Model Initialization ---
        # Choose a Gemini model that supports function calling (e.g., 1.5 Flash/Pro)
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash-latest', # Or 'gemini-1.5-pro-latest'
            # Safety settings might need adjustment depending on tool outputs
            # safety_settings=[ ... ],
            generation_config=GenerationConfig(
                # Adjust temperature if needed (0.0 = deterministic, 1.0 = creative)
                temperature=0.7
            )
        )
        self.chat_session = None # Will hold the Gemini chat session
        # --- End Gemini Model Initialization ---
        self.available_gemini_tools: Optional[List[GeminiTool]] = None


    async def connect_to_server(self, server_script_path: str):
        """Connects to an MCP server via stdio."""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node" # Basic check, might need python3 on Linux/Mac
        print(f"Attempting to start server with: {command} {server_script_path}")
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None # Inherit environment (ensure server has necessary keys if needed)
        )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            await self.session.initialize()

            # List available tools from MCP Server
            response = await self.session.list_tools()
            mcp_tools = response.tools

            # Convert MCP tools to Gemini format
            gemini_func_declarations = [convert_mcp_tool_to_gemini(tool) for tool in mcp_tools]

            if gemini_func_declarations:
                self.available_gemini_tools = [GeminiTool(function_declarations=gemini_func_declarations)]
                print("\nConnected to server. Available tools for Gemini:")
                for func in gemini_func_declarations:
                     print(f"- {func.name}: {func.description}")
            else:
                 print("\nConnected to server, but no tools reported.")
                 self.available_gemini_tools = None

            # --- Initialize Gemini Chat Session ---
            # Start a chat session. History will be managed automatically.
            self.chat_session = self.model.start_chat(enable_automatic_function_calling=False) # We handle calls manually via MCP
            # --- End Initialize Gemini Chat Session ---

        except Exception as e:
            print(f"\nError connecting to or initializing server: {e}")
            await self.cleanup() # Attempt cleanup on connection error
            raise # Re-raise the exception


    async def process_query(self, query: str) -> str:
        """Processes a query using Gemini and available MCP tools."""
        if not self.session:
            return "Error: Not connected to an MCP server."
        if not self.chat_session:
            return "Error: Gemini chat session not initialized."

        print("\nSending query to Gemini...")
        final_text_parts = []

        try:
            # Send the user query to Gemini, providing the tools definition
            response = await self.chat_session.send_message_async(
                query,
                tools=self.available_gemini_tools # Pass available tools
            )

            # --- Gemini Function Calling Loop ---
            while response.candidates[0].content.parts[0].function_call.name:
                function_call = response.candidates[0].content.parts[0].function_call
                tool_name = function_call.name
                # Ensure args are properly converted to dict if needed (might already be dict)
                tool_args = dict(function_call.args)

                print(f"\nGemini requests tool call: {tool_name}({json.dumps(tool_args)})")
                final_text_parts.append(f"[Gemini requested tool '{tool_name}' with arguments: {json.dumps(tool_args)}]")

                # Execute the tool call via MCP
                try:
                    print(f"Executing MCP tool: {tool_name}...")
                    mcp_result = await self.session.call_tool(tool_name, tool_args)
                    print(">>> DEBUG: call_tool successful")
                    print(f">>> DEBUG: mcp_result.content type is {type(mcp_result.content)}")
                    print(f">>> DEBUG: mcp_result.content value is {mcp_result.content}") # Will show list[TextContent]

                    # --- FIX: Extract JSON string from TextContent and parse ---
                    extracted_data = None
                    json_string = None

                    # Check if content is a list containing one TextContent object
                    if (isinstance(mcp_result.content, list) and
                            len(mcp_result.content) == 1 and
                            # Check type by name string to avoid potential import issues/circular deps
                            type(mcp_result.content[0]).__name__ == 'TextContent' and
                            hasattr(mcp_result.content[0], 'text')):

                        json_string = mcp_result.content[0].text
                        print(">>> DEBUG: Extracted text content from TextContent object.")
                    elif mcp_result.content is not None:
                         # Handle other potential content types if needed, or log a warning
                         print(f">>> WARNING: Unexpected content type received: {type(mcp_result.content)}. Attempting to use as is.")
                         # Try to use the content directly if it's already serializable
                         if isinstance(mcp_result.content, (dict, list, str, int, float, bool)):
                             extracted_data = mcp_result.content
                         else:
                             print(f">>> ERROR: Cannot directly serialize content of type {type(mcp_result.content)}. Setting extracted_data to None.")
                             extracted_data = {"error": f"Received unserializable content type {type(mcp_result.content)} from tool."}

                    # If we extracted a JSON string, try to parse it
                    if json_string:
                        try:
                            extracted_data = json.loads(json_string)
                            print(">>> DEBUG: Successfully parsed JSON string into Python object.")
                        except json.JSONDecodeError as json_err:
                            print(f">>> WARNING: Failed to parse JSON: {json_err}. Passing raw string back.")
                            # If JSON is invalid, send the raw string back to Gemini, it might understand
                            extracted_data = json_string
                    # --- END FIX ---

                    # Use the extracted_data (should be dict/list/string or None/error dict)
                    api_response = {"content": extracted_data}
                    print(f">>> DEBUG: api_response prepared: {api_response}")


                    # Create the FunctionResponse part - THIS should now work
                    tool_response_part = genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response=api_response
                        )
                    )
                    print(">>> DEBUG: tool_response_part created successfully.")


                    print("Sending tool result back to Gemini...")
                    response = await self.chat_session.send_message_async(
                        tool_response_part,
                        tools=self.available_gemini_tools
                    )
                    print(">>> DEBUG: send_message_async successful")

                except Exception as tool_error:
                    # This should now only catch errors in the parsing logic above,
                    # or if send_message_async fails for other reasons (network, API key etc.)
                    print(f"Error during tool result processing or sending: {tool_error}")
                    final_text_parts.append(f"[Error processing tool result '{tool_name}': {tool_error}]")

                    # Send a clearer error back to Gemini
                    error_response_part = genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={"error": f"Client failed to process tool result or send it to Gemini: {str(tool_error)}"}
                        )
                    )
                    print("Sending error notification back to Gemini...")
                    try:
                         response = await self.chat_session.send_message_async(
                             error_response_part,
                             tools=self.available_gemini_tools
                         )
                    except Exception as send_error:
                         print(f"Failed to send error back to Gemini: {send_error}")
                         final_text_parts.append("[Failed to inform Gemini about the tool execution error.]")
                         break # Exit the tool call loop

            # --- End Gemini Function Calling Loop ---

            # After potentially handling tool calls, get the final text response
            final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            final_text_parts.append(final_text)
            print("\nGemini final response received.")

        except Exception as e:
            print(f"\nError during Gemini interaction: {e}")
            # You might want more specific error handling here based on Gemini API errors
            # e.g., check for google.api_core.exceptions.PermissionDenied, QuotaExceeded, etc.
            return f"Error processing query with Gemini: {str(e)}"

        return "\n".join(final_text_parts)


    async def chat_loop(self):
        """Runs an interactive chat loop."""
        print("\n--- MCP Client with Gemini Started! ---")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue
                if query.lower() == 'quit':
                    print("Exiting...")
                    break

                response = await self.process_query(query)
                print("\nResponse:")
                print(response)

            except KeyboardInterrupt:
                 print("\nExiting...")
                 break
            except Exception as e:
                # Catch exceptions from process_query or input issues
                print(f"\nAn unexpected error occurred: {str(e)}")
                # Optionally, try to recover or just break
                # break


    async def cleanup(self):
        """Cleans up resources."""
        print("\nCleaning up resources...")
        await self.exit_stack.aclose()
        print("Resources cleaned up.")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script.[py|js]>")
        sys.exit(1)

    server_path = sys.argv[1]
    client = MCPClient()
    try:
        await client.connect_to_server(server_path)
        await client.chat_loop()
    except Exception as e:
        # Catch potential errors during connection or chat loop setup
        print(f"\nFatal error: {e}")
        print("Ensure the server path is correct and the server can run.")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nApplication terminated with error: {e}")