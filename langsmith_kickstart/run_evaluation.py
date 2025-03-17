import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.smith import RunEvalConfig
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def run_evaluation():
    client = Client()
    llm = ChatOpenAI(
        model="llama3.2",
        openai_api_base="http://localhost:11434/v1",
        openai_api_key="ollama",
        temperature=0,
    )
    
    # Configuration de la chaîne QA
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant expert qui répond aux questions de manière précise et concise."),
        ("human", "Question: {question}"),
        ("assistant", "Je vais répondre à votre question de manière claire et structurée.")
    ])
    
    # Création de la chaîne avec le nouveau format RunnableSequence
    chain = qa_prompt | llm | StrOutputParser()
    
    # Configuration de l'évaluation
    eval_config = RunEvalConfig(
        evaluators=[
            "correctness",
            "helpfulness",
            "relevance"
        ]
    )
    
    # Exécution de l'évaluation
    results = client.run_on_dataset(
        dataset_name="evaluation_dataset_prod",
        llm_or_chain_factory=chain,
        evaluation=eval_config,
        project_name="qa_evaluation_prod"
    )
    
    # Vérification des résultats
    feedbacks = client.list_feedback(project_name="qa_evaluation_prod")
    scores = [f.score for f in feedbacks]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    if avg_score < 0.8:
        raise Exception(f"Score moyen trop bas: {avg_score:.2f}")
    
    return results

if __name__ == "__main__":
    run_evaluation() 