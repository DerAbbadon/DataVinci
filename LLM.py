import ollama

question = 'Why is the sky blue?'

response = ollama.chat(model='llama2', messages=[
  {
    'role': 'user',
    'content': question,
  },
])

print(response['message']['content'])