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

        api_key = os.environ["MISTRAL_API_KEY"]
        self.mistral_model = "mistral-large-latest"
        self.client = Mistral(api_key=api_key)


        # Paths to input and output files
        self.output_dir = 'output'
        self.input_json_path = 'input_book.json'
        self.toc_prompt_json_path = 'toc_prompt.json'
        self.toc_output_path = 'output/toc.json'
        self.book_output_path = 'output/book.json'
        self.h1_output_path = 'output/h1.json'
        self.h2_output_path = 'output/h2.json'
        self.h3_output_path = 'output/h3.json'
        self.h4_output_path = 'output/h4.json'

        # Read input data and prompts
        self.book_title, self.book_keywords = self.read_input_json(self.input_json_path)
        self.system_prompt, self.original_prompts = self.read_prompts_json(self.toc_prompt_json_path)

        console.print(f'Book title: {self.book_title}', style="red") 
        console.print(f'Book keywords: {self.book_keywords}', style="yellow")
        console.print(f'System JSON prompt: {self.system_prompt}', style="purple")
        console.print(f'Original JSON prompts: {self.original_prompts}', style="green")


    def generate_table_of_contents(self):
        """Generate the table of contents as a hierarchical JSON structure."""
        toc = {'title': self.book_title, 'headings': []}

        print('\n' + '-' * 100 + '\n')
        print('Generating TOC ...')

        if not os.path.exists(self.h1_output_path):

            # Generate H1 headings
            h1_prompt = self.original_prompts['H1'].format(book_title=self.book_title, book_keywords=self.book_keywords)
            #console.print(f'h1_prompt: {h1_prompt}', style="purple")
            
            h1_response = self.call_mistral_model(h1_prompt)

            h1_response = h1_response.replace("```json", "").replace("```", "").strip()
            h1_response = h1_response.replace("\n", "").replace("\\", "")
            print(f'h1_response: {h1_response}')
            self.write_json_to_file(h1_response, self.h1_output_path)
        
            h1_headings = json.loads(h1_response)['chapters']
            print(f'h1_headings: {h1_headings}')


        self.generate_h2_headings()
        self.generate_h3_headings()
        self.generate_h4_headings()


    def generate_h2_headings(self):
        
        if os.path.exists(self.h1_output_path):
            console.print(f'h1_output_path already exists: {self.h1_output_path}', style="blue")
            if os.path.exists(self.h2_output_path): 
                console.print(f'h2_output_path already exists: {self.h2_output_path}', style="blue")
            else:
                console.print(f'Generating H2 headings...', style="blue")    
                self.h1_response = json.loads(open(self.h1_output_path, 'r', encoding='utf-8').read())
                key1 = list(self.h1_response.keys())[0]
                self.h1_headings = self.h1_response[key1]
                print(f'h1_headings: {self.h1_headings}')
                toc = []
            
                for h1 in self.h1_headings:
                    h1_dict = {'title': h1['title'], 'summary': h1['summary'], 'keywords': h1['keywords']}
                    
                    # Generate H2 headings under H1
                    h2_prompt = self.original_prompts['H2'].format(
                        chapter_title=h1['title'],
                        keywords=h1['keywords']
                    )
                    print(f'h2_prompt: {h2_prompt}')
                    #input("Press ENTER to continue...")

                    self.h2_response = self.call_mistral_model(h2_prompt)
                    self.h2_response = self.h2_response.replace("```json", "").replace("```", "").strip()
                    self.h2_response = self.h2_response.replace("\n", "").replace("\\", "").replace("\'", "\"")
                    self.h2_response = json.loads(self.h2_response)
                    
                    h2_key = list(self.h2_response.keys())[0]
                    for h2 in self.h2_response[h2_key]:
                        h2_dict = {'title': h2['title'], 'summary': h2['summary'], 'keywords': h2['keywords']}
                        toc.append(h2_dict)

                    print(f'h2_response: {self.h2_response}')
                    self.write_json_to_file(self.h2_response, os.path.join(self.output_dir, 'h2_' + str(h1['title']) + '.json'))
                    
                toc_json = '{ "chapters": ' + json.dumps(toc, ensure_ascii=True, indent=4) + '}'
                self.write_json_to_file(toc_json, os.path.join(self.h2_output_path))

        else:
            console.print(f'h1_output_path does not exist: {self.h1_output_path}', style="blue")
            exit()


    def generate_h3_headings(self):

        if os.path.exists(self.h2_output_path): 
            console.print(f'h2_output_path already exists: {self.h2_output_path}', style="blue")
            if os.path.exists(self.h3_output_path):
                console.print(f'h3_output_path already exists: {self.h3_output_path}', style="blue")
            else:
                console.print(f'Generating H3 headings...', style="blue")
                self.h2_response = json.loads(open(self.h2_output_path, 'r', encoding='utf-8').read())
                key1 = list(self.h2_response.keys())[0]
                self.h2_headings = self.h2_response[key1]
                print(f'h2_headings: {self.h2_headings}')
                toc = []
                
                for h2 in self.h2_headings:
                    h2_dict = {'title': h2['title'], 'summary': h2['summary'], 'keywords': h2['keywords']}
                    
                    # Generate H3 headings under H2
                    h3_prompt = self.original_prompts['H3'].format(
                    section_title=h2['title'],
                        keywords=h2['keywords']
                    )
                    print(f'h3_prompt: {h3_prompt}')
                    #input("Press ENTER to continue...")

                    self.h3_response = self.call_mistral_model(h3_prompt)
                    self.h3_response = self.h3_response.replace("```json", "").replace("```", "").strip()
                    self.h3_response = self.h3_response.replace("\n", "").replace("\\", "")
                    self.h3_response = json.loads(self.h3_response)
                    
                    h3_key = list(self.h3_response.keys())[0]
                    for h3 in self.h3_response[h3_key]:
                        h3_dict = {'title': h3['title'], 'summary': h3['summary'], 'keywords': h3['keywords']}
                        toc.append(h3_dict)

                    print(f'h3_response: {self.h3_response}')
                    self.write_json_to_file(self.h3_response, os.path.join(self.output_dir, 'h3_' + str(h2['title']) + '.json'))
                    
                toc_json = '{ "chapters": ' + json.dumps(toc, ensure_ascii=True, indent=4) + '}'
                self.write_json_to_file(toc_json, os.path.join(self.h3_output_path))

        else:
            console.print(f'h2_output_path does not exist: {self.h2_output_path}', style="blue")
            exit()


    def generate_h4_headings(self):

        if os.path.exists(self.h3_output_path):
            console.print(f'h3_output_path already exists: {self.h3_output_path}', style="blue")
            if os.path.exists(self.h4_output_path):
                console.print(f'h4_output_path already exists: {self.h4_output_path}', style="blue")
            else:
                console.print(f'Generating H4 headings...', style="blue")
                self.h3_response = json.loads(open(self.h3_output_path, 'r', encoding='utf-8').read())
                key1 = list(self.h3_response.keys())[0]
                self.h3_headings = self.h3_response[key1]
                print(f'h3_headings: {self.h3_headings}')
                toc = []

                for h3 in self.h3_headings:
                    h3_dict = {'title': h3['title'], 'summary': h3['summary'], 'keywords': h3['keywords']}

                    # Generate H4 headings under H3
                    h4_prompt = self.original_prompts['H4'].format(
                        subsection_title=h3['title'],
                        keywords=h3['keywords']
                        )
                    print(f'h4_prompt: {h4_prompt}')
                    #input("Press ENTER to continue...")

                    self.h4_response = self.call_mistral_model(h4_prompt)
                    self.h4_response = self.h4_response.replace("```json", "").replace("```", "").strip()
                    self.h4_response = self.h4_response.replace("\n", "").replace("\\", "")
                    self.h4_response = json.loads(self.h4_response)

                    h4_key = list(self.h4_response.keys())[0]
                    for h4 in self.h4_response[h4_key]:
                        h4_dict = {'title': h4['title'], 'summary': h4['summary'], 'keywords': h4['keywords']}
                        toc.append(h4_dict)

                    print(f'h4_response: {self.h4_response}')   
                    self.write_json_to_file(self.h4_response, os.path.join(self.output_dir, 'h4_' + str(h3['title']) + '.json'))
                
                toc_json = '{ "chapters": ' + json.dumps(toc, ensure_ascii=True, indent=4) + '}'
                toc_json = toc_json.replace("\n", "").replace("\\", "")
                self.write_json_to_file(toc_json, os.path.join(self.h4_output_path))

        else:
            console.print(f'h3_output_path does not exist: {self.h3_output_path}', style="blue")
            exit()
            

    def call_mistral_model(self, prompt): 
        console.print(f'System prompt: {self.system_prompt}', style="purple")
        console.print(f'AI actual prompt: {prompt}', style="purple")

        
        chat_response = self.client.chat.complete(
            model= self.mistral_model,
            messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": prompt
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

    def write_json_to_file(self, data, file_path, overwrite=False):
        """Write JSON data to a file."""
        print(f'overwrite: {overwrite}')
        print(f'file_path: {file_path}')
        print(f'os.path.dirname(file_path): {os.path.dirname(file_path)}')
        print(f'data: {data}')
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if overwrite:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        else:
            with open(file_path, 'a', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

    def generate_book_content(self):
        """Generate the book content based on the table of contents."""
        for h1 in toc['headings']:
            # Generate content for H1
            h1_prompt = system_prompt + "\n" + prompts['H1'].format(keywords=h1['summary'])
            h1_content = self.call_mistral_model(h1_prompt)
            h1['content'] = h1_content

            for h2 in h1['headings']:
                # Generate content for H2
                h2_prompt = system_prompt + "\n" + prompts['H2'].format(
                    chapter_title=h1['title'],
                    keywords=h2['summary']
                )
                h2_content = self.call_mistral_model(h2_prompt)
                h2['content'] = h2_content

                for h3 in h2['headings']:
                    # Generate content for H3
                    h3_prompt = system_prompt + "\n" + prompts['H3'].format(
                        section_title=h2['title'],
                        keywords=h3['summary']
                    )
                    h3_content = self.call_mistral_model(h3_prompt)
                    h3['content'] = h3_content

                    for h4 in h3['headings']:
                        # Generate content for H4
                        h4_prompt = system_prompt + "\n" + prompts['H4'].format(
                            subsection_title=h3['title'],
                            keywords=h4['summary']
                        )
                        h4_content = self.call_mistral_model(h4_prompt)
                        h4['content'] = h4_content
        return toc

    def parse_response(self, keyword, response):
        try:
            data = json.loads(response)
            return data.get(keyword, [])
        except json.JSONDecodeError:
            print("Error: The response is not a valid JSON format.")
            return []

    def parse_sub_response(self, keyword, response):
        try:
            data = json.loads(response[keyword])
            print(f'data: {data}')
            return data
        except json.JSONDecodeError:
            print("Error: The response is not a valid JSON format.")
            return []

    def main(self):
        

        # Generate table of contents
        self.generate_table_of_contents()
        exit()
        # Generate book content
        book = self.generate_book_content(toc, system_prompt, prompts)
        self.write_json_to_file(book, self.book_output_path)


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


if __name__ == '__main__':
    book_generator = BookGenerator()
    book_generator.main()

