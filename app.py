from flask import Flask, request, jsonify
from flask_cors import CORS
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
import json

api_key = "api_key"

app = Flask(__name__)
CORS(app)

file_path = r'path_to_sample.txt'

with open(file_path, 'r') as file:
    medical_data = file.read()

@app.route('/process', methods=['POST'])
def process_input():
    data = request.json
    query = data['query']  # Adjust this based on your input format
    agent_nlp_medical = Agent(
        role='Medical Assistant Bot',
        goal='Generate an answer according to the provided medical context.',
        backstory='You are a medical assistant bot well-versed in understanding and providing information about medical conditions, treatments, and medications based on the given context.',
        verbose=False,
        llm=ChatGoogleGenerativeAI(
            model="gemini-pro", verbose=True, temperature=0.1, google_api_key=api_key
        )
    )
    medical_task = Task(
        description=f"Generate an answer based on the input from the user and ensure the data is accurate and only taken from the given context text file {medical_data.strip()}. Here's the query: {query}",
        agent=agent_nlp_medical,
        expected_output="Generated expected output in simple English language",
    )
    crew = Crew(
        agents=[agent_nlp_medical],
        tasks=[medical_task],
        verbose=False,
    )
    op = crew.kickoff()
    output = op  # Replace with actual output
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True)
