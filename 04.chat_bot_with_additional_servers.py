import asyncio
import json
from contextlib import AsyncExitStack
from typing import Dict, List, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class MCP_ChatBot:

    def __init__(self):
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server and register its tools."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)

            response = await session.list_tools()
            print(f"Connected to '{server_name}' with tools: {[t.name for t in response.tools]}")

            for tool in response.tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        except Exception as e:
            print(f"Failed to connect to '{server_name}': {e}")

    async def connect_to_servers(self) -> None:
        """Read server_config.json and connect to all configured servers."""
        try:
            with open("server_config.json", "r") as f:
                data = json.load(f)

            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def process_query(self, query: str) -> None:
        messages = [{"role": "user", "content": query}]

        response = self.anthropic.messages.create(
            max_tokens=4096,
            model="claude-haiku-4-5-20251001",
            tools=self.available_tools,
            messages=messages,
        )

        while response.stop_reason == "tool_use":
            # Append full assistant turn at once
            messages.append({"role": "assistant", "content": response.content})

            # Print any text alongside the tool call(s)
            for content in response.content:
                if content.type == "text":
                    print(content.text)

            # Collect all tool results into one user message
            tool_results = []
            for content in response.content:
                if content.type == "tool_use":
                    print(f"Calling tool '{content.name}' with args {content.input}")
                    # Route to the correct server session via the tool_to_session map
                    session = self.tool_to_session[content.name]
                    result = await session.call_tool(content.name, arguments=content.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": result.content,
                    })

            messages.append({"role": "user", "content": tool_results})

            response = self.anthropic.messages.create(
                max_tokens=4096,
                model="claude-haiku-4-5-20251001",
                tools=self.available_tools,
                messages=messages,
            )

        # Final text response
        for content in response.content:
            if content.type == "text":
                print(content.text)

    async def chat_loop(self) -> None:
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                if not query:
                    continue
                await self.process_query(query)
                print("\n")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except EOFError:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self) -> None:
        """Close all server connections in reverse order via AsyncExitStack."""
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
