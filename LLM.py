import ollama

question = 'You are an expert in masking semantic types in data. You will be given a column of the database and are required to mask the data entries with different semantic types. Keep the following points in mind. 1. Mask semantic units in strings. 2. Only mask the semantic types specified. 3. Only mask strings, when you are certain they are belonging to the semantic type. 4. You are allowed to correct values, if you are certain they are erroneous and the repaired value will be masked with a semantic type. semantic types: name, country, currency, city, year <Examples> Data Column: John102, Malt109, Spohie893, ; Masked Column: {name(John)}102, {name(Matt)}109, {name(Sophie)}102; Data Column: Ind-674-PRO, US-823-JUN, US-237-JUN, Zim-843-PRO, Eng-781-JUN, Aus-664-PRO, Ind-473-JUN, usa_837, Eng-573-JUN, Zim-392-PRO; Masked Column: {country(Ind)}-674-PRO, {country(US)}-823-JUN, {country(US)}-237-JUN, {country(Zim)}-843-PRO, {country(Eng)}-781-JUN, {country(Aus)}-664-PRO, {country(Ind)}-473-JUN, {country(US)}_837, {country(Eng)}-573-JUN, {country(Zim)}-392-PRO <Task> Data Column:'

question_end = '; Masked Column: '
column = 'Hannover-2024, Hamburf-2023, Berlin-2020, Magdeburg-2021, Berlin-2021, Bremem-2019, Muenchen-1997'
llm_input = question + column + question_end

response = ollama.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': llm_input,
  },
])

print(response['message']['content'])