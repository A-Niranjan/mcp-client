# MCP Client (Python + Gemini) 🐍✨

A command-line interface (CLI) client built with Python to interact with servers implementing the Model Context Protocol (MCP). This client uses Google's Gemini models for natural language understanding and interaction, enabling it to leverage tools exposed by connected MCP servers via function calling.

This client connects to MCP servers using the **stdio** transport mechanism.

## Features

*   **Connect via Stdio:** Interfaces with MCP servers (Python, Node.js, etc.) running locally using standard input/output.
*   **Interactive Chat:** Provides a simple command-line chat interface.
*   **Gemini Powered:** Leverages Google's Gemini models (e.g., `gemini-1.5-flash-latest`) for conversational AI.
*   **MCP Tool Integration:** Discovers tools available on the connected MCP server and makes them available to Gemini.
*   **Function Calling:** Handles Gemini's requests to call server-side tools and returns the results for natural language response generation.
*   **Conversation History:** Maintains context within a chat session.
*   **Error Handling:** Basic handling for connection issues and tool execution errors.

## System Requirements

*   macOS, Windows, or Linux computer
*   Python 3.8+ installed
*   Latest version of `uv` installed (`pip install uv` or see [uv documentation](https://github.com/astral-sh/uv))
*   Google AI API Key

## Setup

1.  **Clone or Create Project:**
    ```bash
    # Option A: Clone if this is in a repository
    # git clone <your-repo-url>
    # cd <your-repo-name>

    # Option B: Create a new project directory
    mkdir mcp-client-gemini
    cd mcp-client-gemini
    ```

2.  **Initialize Environment using `uv`:**
    ```bash
    # Create project structure and virtual environment
    uv init --template basic . # Use basic template
    uv venv

    # Activate virtual environment
    # On Windows:
    .venv\Scripts\activate
    # On Unix or MacOS:
    source .venv/bin/activate

    # Install required packages
    uv add mcp google-generativeai python-dotenv

    # (Optional) Remove boilerplate if you used `uv init` without --template basic
    # On Windows: del src\mcp_client_gemini\__init__.py src\mcp_client_gemini\main.py tests\__init__.py tests\test_main.py
    # On Unix or MacOS: rm src/mcp_client_gemini/__init__.py src/mcp_client_gemini/main.py tests/__init__.py tests/test_main.py

    # Create/Ensure our main client file exists
    # If using 'uv init --template basic .', rename main.py to client.py or adjust as needed
    # touch client.py
    ```
    *(Make sure your primary code is in a file named `client.py` in the root directory, or adjust the run command accordingly.)*

## Configuration: API Key

You need a Google AI API key to use Gemini.

1.  **Get your Key:** Obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

2.  **Create `.env` File:** In the root of your project directory (`mcp-client-gemini`), create a file named `.env`:
    ```bash
    # On Windows (PowerShell):
    New-Item .env

    # On Unix or MacOS:
    touch .env
    ```

3.  **Add Key to `.env`:** Open the `.env` file and add your API key:
    ```dotenv
    GOOGLE_API_KEY=<your_google_ai_api_key_here>
    ```

4.  **Secure Your Key:** Add `.env` to your `.gitignore` file to prevent accidentally committing your secret key:
    ```bash
    echo ".env" >> .gitignore
    ```

    **Warning:** Keep your API key secure and private!

## Usage

Ensure your virtual environment is activated (`source .venv/bin/activate` or `.venv\Scripts\activate`).

Run the client from your project's root directory, passing the path to the MCP server script as a command-line argument:

```bash
# Example connecting to a Python MCP server
uv run client.py path/to/your/server.py

# Example connecting to a Node.js MCP server (use the built .js file)
uv run client.py path/to/your/build/index.js

# Example using an absolute path (Windows)
uv run client.py C:/Projects/mcp-servers/my-server/server.py

# Example using an absolute path (Linux/MacOS)
uv run client.py /home/user/projects/mcp-servers/my-server/server.py