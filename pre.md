import asyncio
import os
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
import json

# MCP Imports
from mcp import ClientSession, StdioServerParameters, Tool as McpTool  # Renamed McpTool to avoid name clash
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
    gemini_params = {
        "type": "object",
        "properties": mcp_tool.inputSchema.get("properties", {}),
    }

    if "required" in mcp_tool.inputSchema:
        required_fields = mcp_tool.inputSchema["required"]
        if isinstance(required_fields, list) and required_fields:
            gemini_params["required"] = required_fields
        elif isinstance(required_fields, list) and not required_fields:
            pass
        else:
            print(f"Warning: 'required' field in schema for tool '{mcp_tool.name}' is not a list, skipping.")

    func_decl = FunctionDeclaration(
        name=mcp_tool.name,
        description=mcp_tool.description or f"Tool named {mcp_tool.name}",
        parameters=gemini_params
    )
    return func_decl

class MCPClient:
    def __init__(self):
        """Initializes the MCP Client."""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # --- Gemini Model Initialization ---
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=GenerationConfig(
                temperature=0.7
            )
        )
        self.chat_session = None
        # --- End Gemini Model Initialization ---
        self.available_gemini_tools: Optional[List[GeminiTool]] = None

    async def connect_to_server(self, server_script_path: str):
        """Connects to an MCP server via stdio."""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        print(f"Attempting to start server with: {command} {server_script_path}")
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
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

            # Initialize Gemini Chat Session
            self.chat_session = self.model.start_chat(enable_automatic_function_calling=False)
        except Exception as e:
            print(f"\nError connecting to or initializing server: {e}")
            await self.cleanup()
            raise

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
                tools=self.available_gemini_tools
            )

            # --- Gemini Function Calling Loop ---
            while True:
                # Manually serialize the parts to avoid DESCRIPTOR issue
                content = response.candidates[0].content
                parts = content.parts  # This is a RepeatedComposite object
                serializable_parts = []
                function_call = None
                function_call_part = None

                # Convert each part to a dictionary
                for part in parts:
                    part_dict = {}
                    if hasattr(part, 'text') and part.text:
                        part_dict['text'] = part.text
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        function_call_part = part
                        part_dict['function_call'] = {
                            'name': function_call.name,
                            'args': dict(function_call.args)  # Convert MapComposite to dict
                        }
                    serializable_parts.append(part_dict)

                print(f">>> DEBUG: Serializable parts: {serializable_parts}")

                # Check if there's a function call to process
                if not function_call:
                    break  # Exit the loop if no function call is found

                tool_name = function_call.name
                tool_args = dict(function_call.args)

                print(f"\nGemini requests tool call: {tool_name}({json.dumps(tool_args)})")
                final_text_parts.append(f"[Gemini requested tool '{tool_name}' with arguments: {json.dumps(tool_args)}]")

                # Execute the tool call via MCP
                try:
                    print(f"Executing MCP tool: {tool_name}...")
                    mcp_result = await self.session.call_tool(tool_name, tool_args)
                    print(">>> DEBUG: call_tool successful")
                    print(f">>> DEBUG: mcp_result.content type is {type(mcp_result.content)}")
                    print(f">>> DEBUG: mcp_result.content value is {mcp_result.content}")

                    # Extract the text content from mcp_result.content
                    extracted_text = None
                    if (isinstance(mcp_result.content, list) and
                            len(mcp_result.content) == 1 and
                            type(mcp_result.content[0]).__name__ == 'TextContent' and
                            hasattr(mcp_result.content[0], 'text')):
                        extracted_text = mcp_result.content[0].text
                        print(">>> DEBUG: Extracted text content from TextContent object.")
                    else:
                        print(f">>> WARNING: Unexpected content type received: {type(mcp_result.content)}")
                        extracted_text = str(mcp_result.content)

                    # Send the result as a text part
                    tool_response_part = genai.protos.Part(
                        text=f"[Tool '{tool_name}' result]: {extracted_text}"
                    )
                    print(">>> DEBUG: tool_response_part created as text part.")

                    print("Sending tool result back to Gemini...")
                    response = await self.chat_session.send_message_async(
                        tool_response_part,
                        tools=self.available_gemini_tools
                    )
                    print(">>> DEBUG: send_message_async successful")

                except Exception as tool_error:
                    print(f"Error during tool execution or sending: {tool_error}")
                    final_text_parts.append(f"[Error executing tool '{tool_name}': {tool_error}]")

                    # Send an error back to Gemini as a text part
                    error_response_part = genai.protos.Part(
                        text=f"[Error executing tool '{tool_name}']: {str(tool_error)}"
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
                        break

            # After handling tool calls, get the final text response
            final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            final_text_parts.append(final_text)
            print("\nGemini final response received.")

        except Exception as e:
            print(f"\nError during Gemini interaction: {e}")
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
                print(f"\nAn unexpected error occurred: {str(e)}")

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
        print(f"\nFatal error: {e}")
        print("Ensure the server path is correct and the server can run.")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nApplication terminated with error: {e}")