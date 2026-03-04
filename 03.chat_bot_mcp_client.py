import asyncio
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List

load_dotenv()


class MCP_ChatBot:

    def __init__(self):
        self.session: ClientSession = None
        self.anthropic = Anthropic()
        self.available_tools: List[dict] = []

    async def process_query(self, query):
        messages = [{'role': 'user', 'content': query}]

        response = self.anthropic.messages.create(
            max_tokens=2024,
            model='claude-haiku-4-5-20251001',
            tools=self.available_tools,
            messages=messages
        )

        while response.stop_reason == 'tool_use':
            # Append full assistant turn at once
            messages.append({'role': 'assistant', 'content': response.content})

            # Print any text alongside the tool call
            for content in response.content:
                if content.type == 'text':
                    print(content.text)

            # Collect all tool results into one user message
            tool_results = []
            for content in response.content:
                if content.type == 'tool_use':
                    print(f"Calling tool {content.name} with args {content.input}")
                    # Tool is executed by the MCP server via the client session
                    result = await self.session.call_tool(content.name, arguments=content.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": result.content
                    })

            messages.append({"role": "user", "content": tool_results})

            response = self.anthropic.messages.create(
                max_tokens=2024,
                model='claude-haiku-4-5-20251001',
                tools=self.available_tools,
                messages=messages
            )

        # Final text response
        for content in response.content:
            if content.type == 'text':
                print(content.text)

    async def chat_loop(self):
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
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

    async def connect_to_server_and_run(self):
        server_script = os.path.join(os.path.dirname(__file__), "02.chat_bot_mcp_server.py")
        server_params = StdioServerParameters(
            command="python3",
            args=[server_script],
            env=None,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()

                response = await session.list_tools()
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])

                self.available_tools = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in tools]

                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())
