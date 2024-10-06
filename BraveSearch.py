import requests

from rich import print
from rich.console import Console
from rich.text import Text

console = Console()

def main():
    
    api_key = 'BSAL848jGKjzDxeQvDiAReMoCP4DQp8'
    query = 'brave search'
    url = 'https://api.search.brave.com/res/v1/web/search'

    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': api_key,
    }

    params = {
        'q': query,
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f'Failed to get data, status code: {response.status_code}')
        print(response.text)

if __name__ == '__main__':
    main()

