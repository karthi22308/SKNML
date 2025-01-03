import openai

# Set the Azure OpenAI API endpoint and API key
openai.api_base = 'https://hexavarsity-secureapi.azurewebsites.net/api/azureai'  # Replace with your Azure endpoint
openai.api_key = '4ceeaa9071277c5b'  # Replace with your Azure API key
openai.version = '2024-06-01'

# Function to generate a response from OpenAI
def generate_response(input):
    response = openai.Completion.create(
        model="gpt-4",  # Specify the model you are using
        prompt=input,  # The input text for the model
        max_tokens=2560,  # Limit the number of tokens in the response
        temperature=0.7,  # Adjust the creativity of the response
        top_p=0.6,  # Control the diversity of the generated responses
        frequency_penalty=0.7  # Adjust the repetition penalty
    )
    return response.choices[0].text.strip()

# Example usage:
input_text = "Please recommend some courses based on my strengths and weaknesses."
response = generate_response(input_text)
print(response)
