import ollama
import pandas as pd

# this example is without usage of langchain only ollama

context='Write your own context or load a file'

question = '''
write your question
'''

prompt = f'{question} : {context}'

# change to llama3 to see the differences
response = ollama.generate(model = 'llama3.2', prompt = prompt)
response = response['response']

print(response)
