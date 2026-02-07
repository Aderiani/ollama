
import ollama
import os

# Define file paths
input_file = "data/input.txt"
output_file = "data/output.txt"

if not os.path.exists("data"):
    print("Data directory does not exist. Creating 'data' directory.")
    exit(1)
    
with open(input_file, 'r') as infile:
    items = infile.read().strip()
    
    
    
promt = f"Categorize the following items into fruits and vegetables:\n{items}\nProvide the output in JSON format with two keys: 'fruits' and 'vegetables'."

try :
    response = ollama.generate(
        model="kimi-k2.5:cloud",
        prompt=promt,
        stream=False
    )
    generated_text = response.get('response', '')
    print("Generated Response:")
    print(generated_text)
    with open(output_file, 'w') as outfile:
        outfile.write(generated_text.strip())
    print("Response written to output file successfully.")
except Exception as e:
    print(f"An error occurred while generating the response: {e}")
    
    with open(output_file, 'w') as outfile:
        outfile.write("Error generating response.")
    exit(1)