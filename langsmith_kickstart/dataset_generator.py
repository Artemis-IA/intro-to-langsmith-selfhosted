import os
from datetime import datetime
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langsmith.utils import LangSmithConflictError

load_dotenv()

def generate_synthetic_dataset():
    client = Client()
    llm = ChatOpenAI(
        model="llama3.2",
        openai_api_base="http://localhost:11434/v1",
        openai_api_key="ollama",
        temperature=0,  # Température à 0 pour des réponses déterministes
    )
    
    # Création d'un dataset synthétique avec des exemples plus structurés
    synthetic_prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant qui génère uniquement du JSON. Ne réponds pas avec du texte, uniquement avec le JSON demandé."),
        ("human", """Génère un exemple de question avec sa réponse attendue sur le sujet suivant: {topic}
        IMPORTANT: Réponds UNIQUEMENT avec le JSON suivant, sans texte avant ou après:
        {{
            "question": "Question claire et précise",
            "expected_answer": "Réponse détaillée",
            "metadata": {{
                "difficulty": "easy/medium/hard",
                "category": "catégorie du sujet",
                "keywords": ["mot-clé1", "mot-clé2"]
            }}
        }}""")
    ])
    
    # Création de la chaîne avec le nouveau format RunnableSequence
    chain = synthetic_prompt | llm | JsonOutputParser()
    
    topics = ["l'histoire de France", "la science", "la technologie", "l'art", "la littérature"]
    dataset_examples = []
    
    for topic in topics:
        for _ in range(3):  # 3 exemples par sujet
            result = chain.invoke({"topic": topic})
            dataset_examples.append(result)
    
    # Création ou mise à jour du dataset dans LangSmith
    dataset_name = f"evaluation_dataset_{datetime.now().strftime('%Y%m%d')}"
    try:
        # Tentative de création du dataset
        dataset = client.create_dataset(
            dataset_name,
            description="Dataset synthétique pour l'évaluation de QA",
            metadata={
                "version": "1.0", 
                "generation_date": datetime.now().isoformat(),
                "model": "llama3.2",
                "model_version": "3.2"
            }
        )
        print(f"Nouveau dataset créé: {dataset_name}")
    except LangSmithConflictError:
        # Si le dataset existe, on le récupère
        print(f"Dataset existant trouvé: {dataset_name}")
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        if not datasets:
            raise Exception(f"Dataset {dataset_name} non trouvé")
        dataset = datasets[0]
    
    # Préparation des exemples pour la création en masse
    examples = [
        {
            "inputs": {"question": example['question']},
            "outputs": {"answer": example['expected_answer']},
            "metadata": example['metadata']
        }
        for example in dataset_examples
    ]
    
    # Création en masse des exemples
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )
    
    return dataset_name, dataset.id

if __name__ == "__main__":
    dataset_name, dataset_id = generate_synthetic_dataset()
    print(f"Dataset mis à jour avec succès: {dataset_name} (ID: {dataset_id})") 