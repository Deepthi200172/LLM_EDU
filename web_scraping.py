import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup, SoupStrainer
from IPython.display import Markdown, display
from typing import List, Dict
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
# Load environment variables first
load_dotenv()

# Configure headers and constants
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}
SYSTEM_PROMPT = """You are an assistant that analyzes the contents of a website 
and provides a short summary, ignoring text that might be navigation related. 
Respond in markdown."""

class WebsiteAnalyzer:
    """Class to analyze and summarize website content."""
    
    def __init__(self, url: str):
        """
        Initialize website analyzer with URL.
        
        Args:
            url: Website URL to analyze
        """
        self.url = url
        self.title = "No title found"
        self.text = ""
        self._soup = None
        self._fetch_and_parse()

    def _fetch_and_parse(self) -> None:
        """Fetch URL content and parse with BeautifulSoup."""
        try:
            response = requests.get(self.url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            
            # Use parser that only looks at the body content
            strainer = SoupStrainer('body')
            self._soup = BeautifulSoup(response.content, 'html.parser', parse_only=strainer)
            
            self._extract_title()
            self._clean_content()
            self._extract_text()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch website: {str(e)}")

    def _extract_title(self) -> None:
        """Extract title from the webpage."""
        title_tag = self._soup.title if self._soup else None
        self.title = title_tag.get_text().strip() if title_tag else "No title found"

    def _clean_content(self) -> None:
        """Remove irrelevant elements from the page."""
        if not self._soup or not self._soup.body:
            return
            
        for element in self._soup.body(["script", "style", "img", "input", "footer", "nav"]):
            element.decompose()

    def _extract_text(self) -> None:
        """Extract and clean main text content."""
        if self._soup and self._soup.body:
            self.text = self._soup.body.get_text(separator="\n", strip=True)
            # Remove excessive whitespace and truncate
            self.text = ' '.join(self.text.split()[:2000])  # Limit to 2000 words

    @property
    def user_prompt(self) -> str:
        """Generate user prompt for LLM."""
        return (
            f"You are looking at a website titled {self.title}\n"
            "The contents of this website is as follows; please provide a short summary "
            "in markdown. If it includes news or announcements, summarize these too.\n\n"
            f"{self.text}"
        )

    @property
    def messages(self) -> List[Dict[str, str]]:
        """Generate messages array for LLM interaction."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.user_prompt}
        ]

    def summarize(self, llm) -> str:
        """
        Generate website summary using provided LLM.
        
        Args:
            llm: Language model instance with invoke() method
            
        Returns:
            str: Generated summary
        """
        try:
            response = llm.invoke(self.messages)
            return response.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def display_summary(self, llm) -> None:
        """Display formatted summary in Jupyter notebook."""
        summary = self.summarize(llm)
        display(Markdown(summary))


analyzer = WebsiteAnalyzer("https://www.ashokleyland.com/")
print(analyzer.display_summary(llm))



























