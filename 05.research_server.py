import arxiv
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP


PAPER_DIR = "papers"

mcp = FastMCP("research")


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.

    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)

    Returns:
        List of paper IDs found in the search
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = client.results(search)

    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "papers_info.json")

    try:
        with open(file_path, "r") as f:
            papers_info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        papers_info[paper.get_short_id()] = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }

    with open(file_path, "w") as f:
        json.dump(papers_info, f, indent=2)

    print(f"Results saved in: {file_path}")
    return paper_ids


@mcp.tool()
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.

    Args:
        paper_id: The ID of the paper to look for

    Returns:
        JSON string with paper information if found, error message if not found
    """
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as f:
                        papers_info = json.load(f)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

    return f"There's no saved information related to paper {paper_id}."


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("papers://folders")
def get_available_folders() -> str:
    """
    List all available topic folders in the papers directory.

    Returns a markdown list of available research topics.
    """
    folders = []
    if os.path.exists(PAPER_DIR):
        for topic_dir in os.listdir(PAPER_DIR):
            topic_path = os.path.join(PAPER_DIR, topic_dir)
            if os.path.isdir(topic_path):
                if os.path.exists(os.path.join(topic_path, "papers_info.json")):
                    folders.append(topic_dir)

    content = "# Available Topics\n\n"
    if folders:
        for folder in sorted(folders):
            content += f"- {folder}\n"
        content += f"\nUse @<topic> to access papers in that topic.\n"
    else:
        content += "No topics found. Search for papers first.\n"

    return content


@mcp.resource("papers://{topic}")
def get_topic_papers(topic: str) -> str:
    """
    Get detailed information about papers on a specific topic.

    Args:
        topic: The research topic to retrieve papers for
    """
    topic_dir = topic.lower().replace(" ", "_")
    papers_file = os.path.join(PAPER_DIR, topic_dir, "papers_info.json")

    if not os.path.exists(papers_file):
        return f"# No papers found for topic: {topic}\n\nTry searching for papers on this topic first."

    try:
        with open(papers_file, "r") as f:
            papers_data = json.load(f)

        content = f"# Papers on {topic.replace('_', ' ').title()}\n\n"
        content += f"Total papers: {len(papers_data)}\n\n"

        for paper_id, info in papers_data.items():
            content += f"## {info['title']}\n"
            content += f"- **Paper ID**: {paper_id}\n"
            content += f"- **Authors**: {', '.join(info['authors'])}\n"
            content += f"- **Published**: {info['published']}\n"
            content += f"- **PDF**: [{info['pdf_url']}]({info['pdf_url']})\n\n"
            content += f"### Summary\n{info['summary'][:500]}...\n\n"
            content += "---\n\n"

        return content
    except json.JSONDecodeError:
        return f"# Error reading papers data for {topic}\n\nThe data file is corrupted."


# ── Prompts ───────────────────────────────────────────────────────────────────

@mcp.prompt()
def generate_search_prompt(topic: str, num_papers: int = 5) -> str:
    """Generate a prompt for Claude to find and discuss academic papers on a specific topic."""
    return f"""Search for {num_papers} academic papers about '{topic}' using the search_papers tool. Follow these instructions:

1. First, search for papers using search_papers(topic='{topic}', max_results={num_papers})
2. For each paper found, extract and organize the following information:
   - Paper title
   - Authors
   - Publication date
   - Brief summary of the key findings
   - Main contributions or innovations
   - Methodologies used
   - Relevance to the topic '{topic}'

3. Provide a comprehensive summary that includes:
   - Overview of the current state of research in '{topic}'
   - Common themes and trends across the papers
   - Key research gaps or areas for future investigation
   - Most impactful or influential papers in this area

4. Organize your findings in a clear, structured format with headings and bullet points for easy readability.

Please present both detailed information about each paper and a high-level synthesis of the research landscape in {topic}."""


if __name__ == "__main__":
    mcp.run(transport='stdio')
