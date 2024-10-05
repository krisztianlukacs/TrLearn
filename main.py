import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

from mistralai import Mistral

from rich import print
from rich.console import Console
from rich.text import Text

console = Console()


class BookGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def read_input_json(self, file_path):
        """Read input JSON file containing the book title and keywords."""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data['book_title'], data['keywords']

    def read_prompts_json(self, file_path):
        """Read JSON file containing prompts for each heading level and the system prompt."""
        with open(file_path, 'r', encoding='utf-8') as file:
            prompts = json.load(file)
        return prompts['system_prompt'], prompts['prompts']

    def generate_table_of_contents(self, book_title, keywords, system_prompt, prompts):
        """Generate the table of contents as a hierarchical JSON structure."""
        toc = {'title': book_title, 'headings': []}

        print('\n' + '-' * 100 + '\n')
        print(f'book_title: {book_title}')
        print(f'keywords: {keywords}')
        print(f'system_prompt: {system_prompt}')
        print(f'prompts: {prompts}')

        # Generate H1 headings
        h1_prompt = system_prompt + "\n" + prompts['H1'].format(keywords=keywords)
        console.print(f'h1_prompt: {h1_prompt}', style="yellow")
        
        h1_response = self.call_mistral_model(h1_prompt)

        h1_response = h1_response.replace("```json", "").replace("```", "").strip()
        print(f'h1_response: {h1_response}')
    
        h1_headings = json.loads(h1_response)
        print(f'h1_headings: {h1_headings}')
        
        exit()
        for h1 in h1_headings:
            h1_dict = {'title': h1['title'], 'summary': h1['summary'], 'headings': []}

            # Generate H2 headings under H1
            h2_prompt = system_prompt + "\n" + prompts['H2'].format(
                chapter_title=h1['title'],
                keywords=keywords
            )
            h2_response = self.call_mistral_model(h2_prompt)
            h2_headings = self.parse_headings_response(h2_response)

            for h2 in h2_headings:
                h2_dict = {'title': h2['title'], 'summary': h2['summary'], 'headings': []}

                # Generate H3 headings under H2
                h3_prompt = system_prompt + "\n" + prompts['H3'].format(
                    section_title=h2['title'],
                    keywords=keywords
                )
                h3_response = self.call_mistral_model(h3_prompt)
                h3_headings = self.parse_headings_response(h3_response)

                for h3 in h3_headings:
                    h3_dict = {'title': h3['title'], 'summary': h3['summary'], 'headings': []}

                    # Generate H4 headings under H3
                    h4_prompt = system_prompt + "\n" + prompts['H4'].format(
                        subsection_title=h3['title'],
                        keywords=keywords
                    )
                    h4_response = self.call_mistral_model(h4_prompt)
                    h4_headings = self.parse_headings_response(h4_response)

                    h3_dict['headings'].extend(h4_headings)
                h2_dict['headings'].append(h3_dict)
            h1_dict['headings'].append(h2_dict)
            toc['headings'].append(h1_dict)

        return toc

    def call_mistral_model(self, prompt): 
        console.print(f'prompt: {prompt}', style="red")

        api_key = os.environ["MISTRAL_API_KEY"]
        mistral_model = "mistral-large-latest"

    

        client = Mistral(api_key=api_key)
        chat_response = client.chat.complete(
            model= mistral_model,
            messages = [
            {
                "role": "system",
                "content": "You are an assistant that helps generate book content. Don't say anything else but the content you are asked to generate. The output should be in JSON format. The chepters should be in H2 format and the sections in H3 format and the subsections in H4 format. " +  
                'Format example: ' +
                '{' +
                ' "chapters": [' +
                    '{' +
                        ' "title": "## Chapter 1: Introduction to Artificial Intelligence",' +
                        ' "summary": "This chapter provides an overview of the field of Artificial Intelligence (AI), its history, and its applications in various industries."' +
                    '},' +
                    '{' +
                        ' "title": "## Chapter 2: Fundamentals of Machine Learning",' +
                            ' "summary": "This chapter introduces the basic concepts of machine learning, including supervised and unsupervised learning, and key algorithms."' +
                    '}' +
                ']' +
                '}'
            },
            {
                "role": "user",
                "content": "Generate a list of chapter titles and brief summaries for a book titled 'artificial intelligence, machine learning, neural networks'. "
            }
            ]
        )
        # print(chat_response.choices[0].message.content)
        
        return chat_response.choices[0].message.content


    def call_mistral_model_original(self, prompt):
        """Generate text using the Mistral AI model."""

        console.print(f'prompt: {prompt}', style="purple")
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=False)
        attention_mask = inputs.ne(tokenizer.pad_token_id).long()
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)
        
        generation_config = GenerationConfig(
            num_beams=4, 
            early_stopping=True,
            max_length=512
        )
        
        output = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=512,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            generation_config=generation_config
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        return response

    def write_json_to_file(self, data, file_path):
        """Write JSON data to a file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def generate_book_content(self, toc, system_prompt, prompts):
        """Generate the book content based on the table of contents."""
        for h1 in toc['headings']:
            # Generate content for H1
            h1_prompt = system_prompt + "\n" + prompts['H1'].format(keywords=h1['summary'])
            h1_content = call_mistral_model(h1_prompt)
            h1['content'] = h1_content

            for h2 in h1['headings']:
                # Generate content for H2
                h2_prompt = system_prompt + "\n" + prompts['H2'].format(
                    chapter_title=h1['title'],
                    keywords=h2['summary']
                )
                h2_content = call_mistral_model(h2_prompt)
                h2['content'] = h2_content

                for h3 in h2['headings']:
                    # Generate content for H3
                    h3_prompt = system_prompt + "\n" + prompts['H3'].format(
                        section_title=h2['title'],
                        keywords=h3['summary']
                    )
                    h3_content = call_mistral_model(h3_prompt)
                    h3['content'] = h3_content

                    for h4 in h3['headings']:
                        # Generate content for H4
                        h4_prompt = system_prompt + "\n" + prompts['H4'].format(
                            subsection_title=h3['title'],
                            keywords=h4['summary']
                        )
                        h4_content = call_mistral_model(h4_prompt)
                        h4['content'] = h4_content
        return toc

    def main(self):
        # Paths to input and output files
        input_json_path = 'input_book.json'
        prompts_json_path = 'prompts.json'
        toc_output_path = 'output/toc.json'
        book_output_path = 'output/book.json'

        # Read input data and prompts
        book_title, keywords = self.read_input_json(input_json_path)
        system_prompt, prompts = self.read_prompts_json(prompts_json_path)

        print(f'book_title: {book_title}')
        print(f'keywords: {keywords}')
        print(f'system_prompt: {system_prompt}')
        print(f'prompts: {prompts}')

        # Generate table of contents
        toc = self.generate_table_of_contents(book_title, keywords, system_prompt, prompts)
        self.write_json_to_file(toc, toc_output_path)

        # Generate book content
        book = self.generate_book_content(toc, system_prompt, prompts)
        self.write_json_to_file(book, book_output_path)

if __name__ == '__main__':
    book_generator = BookGenerator()
    book_generator.main()

