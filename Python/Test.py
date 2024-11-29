import ollama
import pandas as pd

# this example is without usage of langchain only ollama

context='Very manja and gentle stray cat found, we would really like to find a home for it because we cannot keep her for ourselves for long. Has a very cute high pitch but soft meow. Please contact me if you would be interested in adopting.'

question = '''
create a dictionary from the description of this animal ->
'''

prompt = f'{question} : {context}'

# change to llama3 to see the differences
response = ollama.generate(model = 'llama3.2', prompt = prompt)
response = response['response']

print(response)
