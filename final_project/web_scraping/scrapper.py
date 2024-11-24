import requests
from bs4 import BeautifulSoup

# URL of the page to scrape
url = "https://www.geeksforgeeks.org/data-science-for-beginners/"

# Send an HTTP request to the website
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the article title
    title = soup.find('div', class_='article-title').get_text(strip=True)

    # Extract the main content
    content_div = soup.find('div', class_='text')
    
    # Extract text from paragraphs, spans, and blockquotes
    paragraphs = [p.get_text(strip=True) for p in content_div.find_all('p')]
    spans = [span.get_text(strip=True) for span in content_div.find_all('span')]
    blockquotes = [blockquote.get_text(strip=True) for blockquote in content_div.find_all('blockquote')]

    # Extract list items from ordered and unordered lists
    ordered_list_items = []
    unordered_list_items = []

    ordered_lists = content_div.find_all('ol')  # All <ol> tags
    unordered_lists = content_div.find_all('ul')  # All <ul> tags

    for ol in ordered_lists:
        ordered_list_items.extend([li.get_text(strip=True) for li in ol.find_all('li')])
    
    for ul in unordered_lists:
        unordered_list_items.extend([li.get_text(strip=True) for li in ul.find_all('li')])

    # Extract subheadings (e.g., <h2> tags)
    subheadings = [h2.get_text(strip=True) for h2 in content_div.find_all('h2')]

    # Save the extracted content to a text file
    with open("scraped_data.txt", "w", encoding="utf-8") as file:
        # Write title
        file.write(f"Title: {title}\n\n")

        # Write subheadings
        file.write("Subheadings:\n")
        for subheading in subheadings:
            file.write(f"- {subheading}\n")
        file.write("\n")

        # Write paragraphs
        file.write("Paragraphs:\n")
        for paragraph in paragraphs:
            file.write(paragraph + "\n")
        file.write("\n")

        # Write spans
        file.write("Spans:\n")
        for span in spans:
            file.write(span + "\n")
        file.write("\n")

        # Write blockquotes
        file.write("Blockquotes:\n")
        for blockquote in blockquotes:
            file.write(blockquote + "\n")
        file.write("\n")

        # Write ordered list items
        file.write("Ordered List Items:\n")
        for i, item in enumerate(ordered_list_items, 1):
            file.write(f"{i}. {item}\n")
        file.write("\n")

        # Write unordered list items
        file.write("Unordered List Items:\n")
        for item in unordered_list_items:
            file.write(f"- {item}\n")
        file.write("\n")

    print("Scraped data saved to 'scraped_data.txt'")
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
