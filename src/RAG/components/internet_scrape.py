from googlesearch import search
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader
class InternetScraper:
    def __init__(self,config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn",device= self.device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(self.device)

    def get_search_results(self,query, num_results=5, lang="en"):
        """
        Perform a Google search and return the top results.

        Parameters:
        query (str): The search query.
        num_results (int): Number of search results to retrieve. Default is 5.
        lang (str): Language of the search results. Default is "en".

        Returns:
        list: A list of URLs from the search results.
        """
        try:
            urls = [url for url in search(query, num_results=num_results, lang=lang)]
            return urls
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def scrape_webpage(self, url):
        """
        Scrape a webpage to extract its title and paragraph content.

        Parameters:
        url (str): The URL of the webpage to scrape.

        Returns:
        dict: A dictionary containing the title and a list of paragraph content.
        """
        try:
            
            response = requests.get(url)
            print(f"Fetching {url}, Status code: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                title = soup.title.string if soup.title else "No Title Found"
                content = soup.get_text()
                return {"title": title, "content": content[:10000] if len(content)>10000 else content}
            else:
                print(f"Failed to fetch webpage: {url}")
                return {"title": None, "content": ""}

        except Exception as e:
            print(f"An error occurred while scraping {url}: {e}")
            return {"title": None, "content": ""}
    def scrape_urls(self,urls):
        data = [self.scrape_webpage(url) for url in urls if url]
        results = [url_data for url_data in data if url_data["title"] and len(url_data["content"])>500]

        return results




    def summarize_content(self, content, query):
        """
        Summarize a single piece of content by splitting it into chunks and summarizing each chunk.
        """
        # Join content chunks with "---" separator
        content = "---".join(content)
        chunk_size = 1000  # Define the size of each chunk
        content_chunks = [query +"\n  "+content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

        # Create a DataLoader to handle batching (batch size of 4)
        data_loader = DataLoader(content_chunks, batch_size=4)

        summaries = []  # List to store summaries

        # Loop through the batches of content chunks
        for batch_texts in data_loader:
            # Tokenize the batch of texts
            batch = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1020
            )

            # Move the batch to the GPU (if available)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Generate summaries for this batch
            summary_ids = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=50,
                min_length=5,
                num_beams=4,  # Optional beam search
                do_sample=False
            )

            # Decode the summaries and append to the summaries list
            for summary in summary_ids:
                summaries.append(self.tokenizer.decode(summary, skip_special_tokens=True))

        # Join all the summaries together into a final summary
        return " ".join(summaries)


if __name__ == "__main__":
    query = "explain gen ai in detail?"
    obj = InternetScraper(None)
    urls = obj.get_search_results(query, num_results=5)
    
    result = obj.scrape_urls(urls)
    print(len(result))
    print([len(result['content']) for result in result])
    # Summarize the array of content in parallel
    summaries = obj.summarize_content([data['content'] for data in result],"HLo")

    # Print summaries
    print(summaries)
