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

        self.toc_output_dir = 'output/toc'
        self.content_output_dir = 'output/content'
        
        self.input_book_json = 'input_book.json' # book title and keywords
        self.book_data = self.read_json_from_file(self.input_book_json)
        self.book_title = self.book_data['book_title']
        self.book_keywords = self.book_data['keywords']

        self.input_toc_json = 'toc_prompt.json' # table of contents
        self.toc_data = self.read_json_from_file(self.input_toc_json)
        self.toc_system_prompt = self.toc_data['toc_system_prompt']
        self.toc_prompts = self.toc_data['toc_prompts']
        
        self.book_prompt_json = 'book_prompt.json'
        self.book_data = self.read_json_from_file(self.book_prompt_json)
        self.book_content_system_prompt = self.book_data['book_content_system_prompt']
        self.book_content_prompts = self.book_data['book_content_prompts']


        self.content_quality_check_system_json = 'content_quality_check_system_prompt.json'
        self.content_quality_check_data = self.read_json_from_file(self.content_quality_check_system_json)
        self.content_quality_check_system_prompt = self.content_quality_check_data['content_quality_system_prompt']
        
        
        
        self.h1_json = 'h1.json'
        self.h2_json = 'h2.json'
        self.h3_json = 'h3.json'
        self.h4_json = 'h4.json'


        console.print(f'Book title: {self.book_title}', style="red") 
        console.print(f'Book keywords: {self.book_keywords}', style="yellow")
        console.print(f'TOC System JSON prompt: {self.toc_system_prompt}', style="purple")
        console.print(f'TOC JSON prompts: {self.toc_prompts}', style="green")



    def generate_table_of_contents(self):
        """Generate the table of contents as a hierarchical JSON structure."""
        toc = {'title': self.book_title, 'headings': []}

        print('\n' + '-' * 100 + '\n')
        print('Generating TOC ...')

        if not os.path.exists(self.toc_output_dir + '/' + self.h1_json):

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
        
        if os.path.exists(self.toc_output_dir + '/' + self.h1_json):
            console.print(f'h1_output_path already exists: {self.toc_output_dir + '/' + self.h1_json}', style="blue")
            if os.path.exists(self.toc_output_dir + '/' + self.h2_json): 
                console.print(f'h2_output_path already exists: {self.toc_output_dir + '/' + self.h2_json}', style="blue")
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
                    self.write_json_to_file(self.h2_response, os.path.join(self.toc_output_dir, 'h2_' + str(h1['title']) + '.json'))
                    
                toc_json = '{ "chapters": ' + json.dumps(toc, ensure_ascii=True, indent=4) + '}'
                toc_json = toc_json.replace("\n", "").replace("\\", "")
                self.write_json_to_file(toc_json, os.path.join(self.h2_output_path))

        else:
            console.print(f'h1_output_path does not exist: {self.h1_output_path}', style="blue")
            exit()


    def generate_h3_headings(self):

        if os.path.exists(self.toc_output_dir + '/' + self.h2_json): 
            console.print(f'h2_output_path already exists: {self.toc_output_dir + '/' + self.h2_json}', style="blue")
            if os.path.exists(self.toc_output_dir + '/' + self.h3_json):
                console.print(f'h3_output_path already exists: {self.toc_output_dir + '/' + self.h3_json}', style="blue")
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
                    self.write_json_to_file(self.h3_response, os.path.join(self.toc_output_dir, 'h3_' + str(h2['title']) + '.json'))
                    
                toc_json = '{ "chapters": ' + json.dumps(toc, ensure_ascii=True, indent=4) + '}'
                toc_json = toc_json.replace("\n", "").replace("\\", "")
                self.write_json_to_file(toc_json, os.path.join(self.h3_output_path))

        else:
            console.print(f'h2_output_path does not exist: {self.h2_output_path}', style="blue")
            exit()


    def generate_h4_headings(self):

        if os.path.exists(self.toc_output_dir + '/' + self.h3_json):
            console.print(f'h3_output_path already exists: {self.toc_output_dir + '/' + self.h3_json}', style="blue")
            if os.path.exists(self.toc_output_dir + '/' + self.h4_json):
                console.print(f'h4_output_path already exists: {self.toc_output_dir + '/' + self.h4_json}', style="blue")
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
                    self.write_json_to_file(self.h4_response, os.path.join(self.toc_output_dir, 'h4_' + str(h3['title']) + '.json'))
                
                toc_json = '{ "chapters": ' + json.dumps(toc, ensure_ascii=True, indent=4) + '}'
                toc_json = toc_json.replace("\n", "").replace("\\", "")
                self.write_json_to_file(toc_json, os.path.join(self.h4_output_path))

        else:
            console.print(f'h3_output_path does not exist: {self.h3_output_path}', style="blue")
            exit()
            

    def call_mistral_model(self, prompt, phase = 'toc'): 
        if phase == 'toc':
            system_prompt = self.toc_system_prompt
        elif phase == 'book':
            system_prompt = self.book_content_system_prompt
        elif phase == 'quentent_quality_check':
            system_prompt = self.quentent_quality_check_system_prompt

        console.print(f'System prompt: {system_prompt}', style="purple")
        console.print(f'Prompt: {prompt}', style="purple")

        
        chat_response = self.client.chat.complete(
            model= self.mistral_model,
            messages = [
            {
                "role": "system",
                "content": system_prompt
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
        print(f'Generating book content...')

        if os.path.exists(self.toc_output_dir + '/' + self.h4_json):
            self.h4_response = json.loads(open(self.toc_output_dir + '/' + self.h4_json, 'r', encoding='utf-8').read())
            key1 = list(self.h4_response.keys())[0]
            self.h4_headings = self.h4_response[key1]
            #print(f'h4_headings: {self.h4_headings}')


            for h4  in self.h4_headings:
                # Generate content for H1
                h4_dict = {'title': h4['title'], 'summary': h4['summary'], 'keywords': h4['keywords']}
                
                h4_content_prompt = self.book_content_prompts[ 'content'].format(title=h4['title'], summary=h4['summary'], keywords=h4['keywords'])
                print(f'h4_content_prompt: {h4_content_prompt}')

                h4_content = self.call_mistral_model(h4_content_prompt, phase='book')
                h4_content = h4_content.replace("```json", "").replace("```", "").strip()
                h4_content = h4_content.replace("\n", "").replace("\\", "")
                h4_content = json.loads(h4_content)
                print(f'h4_content: {h4_content}')
                content = h4_content['chapters'][0]['content']
                print(f'content: {content}')
                input("Press ENTER to continue...")
                
                h4_summary_prompt = self.book_content_prompts['summary'].format(content=content)
                print(f'h4_summary_prompt: {h4_summary_prompt}')
                
                h4_summary = self.call_mistral_model(h4_summary_prompt, phase='book')
                h4_summary = h4_summary.replace("```json", "").replace("```", "").strip()
                h4_summary = h4_summary.replace("\n", "").replace("\\", "")
                h4_summary = json.loads(h4_summary)
                print(f'h4_summary: {h4_summary}')
                summary = h4_summary['chapters'][0]['summary']
                print(f'summary: {summary}')
                input("Press ENTER to continue...")
                h4_dictionary_prompt = self.book_content_prompts['dictionary'].format(content=content)
                print(f'h4_dictionary_prompt: {h4_dictionary_prompt}')
                h4_dictionary = self.call_mistral_model(h4_dictionary_prompt, phase='book')
                h4_dictionary = h4_dictionary.replace("```json", "").replace("```", "").strip()
                h4_dictionary = h4_dictionary.replace("\n", "").replace("\\", "")
                h4_dictionary = json.loads(h4_dictionary)
                print(f'h4_dictionary: {h4_dictionary}')

                k1 = list(h4_dictionary.keys())[0]
                dictionary = h4_dictionary[k1]
                print(f'dictionary: {dictionary}')
                input("Press ENTER to continue...")

                h4_links_prompt = self.book_content_prompts['links'].format(content=content)
                print(f'h4_links_prompt: {h4_links_prompt}')
                h4_links = self.call_mistral_model(h4_links_prompt, phase='book')
                h4_links = h4_links.replace("```json", "").replace("```", "").strip()   
                h4_links = h4_links.replace("\n", "").replace("\\", "")
                h4_links = json.loads(h4_links)
                print(f'h4_links: {h4_links}')
                links = h4_links['chapters'][0]['links']
                print(f'links: {links}')
                input("Press ENTER to continue...")

                h4['content'] = content
                h4['summary'] = summary
                h4['dictionary'] = dictionary
                h4['links'] = links

                filename = 'h4_' + str(h4['title']) + '.json'
                self.write_json_to_file(h4, os.path.join(self.content_output_dir, filename), overwrite=True)
                print(f'Content generated: {filename} at {self.content_output_dir}')
                input("Press ENTER to continue...")
                

            

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
        # Generate book content
        self.generate_book_content()
        


    def read_json_from_file(self, file_path):
        """Read input JSON file """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    def quentent_quality_check(self, text):
        score = self.call_mistral_model(text, phase='quentent_quality_check')
        return score


if __name__ == '__main__':
    book_generator = BookGenerator()
    book_generator.main()

