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
        # New: resource and prompt registries
        self.resource_to_session: Dict[str, ClientSession] = {}   # uri → session
        self.prompt_to_session: Dict[str, ClientSession] = {}     # name → session
        self.available_prompts: List[dict] = []

    # ── Server connection ─────────────────────────────────────────────────────

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server and register its tools, resources and prompts."""
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

            # ── Tools ──────────────────────────────────────────────────────
            tools_response = await session.list_tools()
            print(f"Connected to '{server_name}' | tools: {[t.name for t in tools_response.tools]}", end="")
            for tool in tools_response.tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })

            # ── Resources ─────────────────────────────────────────────────
            try:
                resources_response = await session.list_resources()
                resource_uris = [r.uri for r in resources_response.resources]
                print(f" | resources: {resource_uris}", end="")
                for resource in resources_response.resources:
                    self.resource_to_session[str(resource.uri)] = session
            except Exception:
                pass  # server may not support resources

            # ── Prompts ───────────────────────────────────────────────────
            try:
                prompts_response = await session.list_prompts()
                prompt_names = [p.name for p in prompts_response.prompts]
                print(f" | prompts: {prompt_names}", end="")
                for prompt in prompts_response.prompts:
                    self.prompt_to_session[prompt.name] = session
                    self.available_prompts.append({
                        "name": prompt.name,
                        "description": prompt.description,
                        "arguments": [
                            {"name": a.name, "description": a.description, "required": a.required}
                            for a in (prompt.arguments or [])
                        ]
                    })
            except Exception:
                pass  # server may not support prompts

            print()  # newline after connection summary

        except Exception as e:
            print(f"\nFailed to connect to '{server_name}': {e}")

    async def connect_to_servers(self) -> None:
        """Read server_config.json and connect to all configured servers."""
        try:
            with open("server_config.json", "r") as f:
                data = json.load(f)
            for server_name, server_config in data.get("mcpServers", {}).items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    # ── Core query processing ─────────────────────────────────────────────────

    async def process_query(self, query: str) -> None:
        messages = [{"role": "user", "content": query}]

        response = self.anthropic.messages.create(
            max_tokens=4096,
            model="claude-haiku-4-5-20251001",
            tools=self.available_tools,
            messages=messages,
        )

        while response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            for content in response.content:
                if content.type == "text":
                    print(content.text)

            tool_results = []
            for content in response.content:
                if content.type == "tool_use":
                    print(f"Calling tool '{content.name}' with args {content.input}")
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

        for content in response.content:
            if content.type == "text":
                print(content.text)

    # ── Resource helpers ──────────────────────────────────────────────────────

    async def get_resource(self, resource_name: str) -> None:
        """
        Fetch a resource by name and print its content.
        Maps @folders → papers://folders and @<topic> → papers://<topic>.
        """
        uri = f"papers://{resource_name}"

        # Find the session: exact match, then fall back to any template match
        session = self.resource_to_session.get(uri)
        if not session:
            for stored_uri, s in self.resource_to_session.items():
                if "{" in stored_uri:   # it's a URI template
                    session = s
                    break

        if not session:
            print(f"No server found that exposes resource '{uri}'.")
            return

        try:
            result = await session.read_resource(uri)
            for content in result.contents:
                if hasattr(content, "text"):
                    print(content.text)
                else:
                    print(content)
        except Exception as e:
            print(f"Error fetching resource '{uri}': {e}")

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def list_prompts(self) -> None:
        """Print all available prompts and their arguments."""
        if not self.available_prompts:
            print("No prompts available.")
            return

        print("\nAvailable prompts:")
        for p in self.available_prompts:
            print(f"  /prompt {p['name']}", end="")
            for arg in p.get("arguments", []):
                req = "" if arg.get("required") else "?"
                print(f" {arg['name']}{req}=<value>", end="")
            print()
            if p.get("description"):
                print(f"    {p['description']}")

    async def execute_prompt(self, prompt_name: str, args: Dict[str, str]) -> None:
        """
        Retrieve a prompt template from the server, render it with args,
        then pass the rendered prompt to the LLM.
        """
        session = self.prompt_to_session.get(prompt_name)
        if not session:
            print(f"Unknown prompt: '{prompt_name}'")
            return

        try:
            result = await session.get_prompt(prompt_name, arguments=args)
        except Exception as e:
            print(f"Error getting prompt '{prompt_name}': {e}")
            return

        # Build the query from the returned messages
        prompt_text = "\n".join(
            msg.content if isinstance(msg.content, str)
            else " ".join(block.text for block in msg.content if hasattr(block, "text"))
            for msg in result.messages
        )

        print(f"\n[Executing prompt '{prompt_name}' with args {args}]\n")
        await self.process_query(prompt_text)

    # ── Chat loop ─────────────────────────────────────────────────────────────

    async def chat_loop(self) -> None:
        print("\nMCP Chatbot Started!")
        print("Commands:")
        print("  @folders             – list available research topics")
        print("  @<topic>             – show papers under that topic")
        print("  /prompts             – list available prompt templates")
        print("  /prompt <name> [k=v] – run a prompt template")
        print("  quit                 – exit\n")

        while True:
            try:
                query = input("Query: ").strip()

                if not query:
                    continue

                if query.lower() == "quit":
                    break

                # ── Resource: @folders or @<topic> ────────────────────────
                elif query.startswith("@"):
                    resource_name = query[1:].strip()
                    await self.get_resource(resource_name)

                # ── List prompts ───────────────────────────────────────────
                elif query == "/prompts":
                    self.list_prompts()

                # ── Execute prompt: /prompt <name> [key=value ...] ─────────
                elif query.startswith("/prompt "):
                    parts = query[len("/prompt "):].split()
                    if not parts:
                        print("Usage: /prompt <name> [key=value ...]")
                        continue
                    prompt_name = parts[0]
                    args = {}
                    for part in parts[1:]:
                        if "=" in part:
                            k, v = part.split("=", 1)
                            args[k] = v
                        else:
                            print(f"Skipping malformed argument '{part}' (expected key=value)")
                    await self.execute_prompt(prompt_name, args)

                # ── Normal LLM query ───────────────────────────────────────
                else:
                    await self.process_query(query)

                print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except EOFError:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def cleanup(self) -> None:
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
