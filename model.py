import google.generativeai as genai

genai.configure(api_key="AIzaSyADtnFwPAMwmNkXgg93FFBFU50xUOC3IAc")

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)