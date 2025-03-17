import os
from datetime import datetime
from dotenv import load_dotenv
from langsmith import Client, RunEvaluator
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.smith import RunEvalConfig
from langchain_core.output_parsers import StrOutputParser
from langsmith.utils import LangSmithConflictError

load_dotenv()

def response_length_evaluator(run):
    """Évalue la longueur de la réponse."""
    response = run.outputs.get("output", "")
    if not response:
        return {"score": 0.0, "reasoning": "Pas de réponse"}
    
    # Score entre 0 et 1 basé sur la longueur
    length = len(response)
    if length < 50:
        score = 0.5
        reasoning = "Réponse trop courte"
    elif length > 500:
        score = 0.5
        reasoning = "Réponse trop longue"
    else:
        score = 1.0
        reasoning = "Longueur appropriée"
    
    return {"score": score, "reasoning": reasoning}

def setup_evaluation(dataset_name, dataset_id):
    client = Client()
    llm = ChatOpenAI(
        model="llama3.2",
        openai_api_base="http://localhost:11434/v1",
        openai_api_key="ollama",
        temperature=0
    )
    
    # Création de la chaîne à évaluer avec un prompt plus sophistiqué
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant expert qui répond aux questions de manière précise et concise."),
        ("human", "Question: {question}"),
        ("assistant", "Je vais répondre à votre question de manière claire et structurée.")
    ])
    
    # Création de la chaîne avec le nouveau format RunnableSequence
    chain = qa_prompt | llm | StrOutputParser()
    
    # Configuration des évaluateurs avec les bons types
    eval_config = RunEvalConfig(
        evaluators=[
            "qa",  # Évaluateur de base pour les questions-réponses
            "criteria",  # Évaluateur basé sur des critères
            "string_distance",  # Évaluateur de distance entre chaînes
            "exact_match"  # Évaluateur de correspondance exacte
        ],
        custom_evaluators=[response_length_evaluator]  # Utilisation de la fonction directement
    )
    
    # Génération d'un nom de projet unique avec timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    project_name = f"qa_evaluation_{timestamp}"
    
    try:
        # Tentative de création du projet
        results = client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=chain,
            evaluation=eval_config,
            project_name=project_name,
            metadata={
                "model": "llama3.2",
                "evaluation_date": datetime.now().isoformat(),
                "dataset_id": dataset_id
            }
        )
    except LangSmithConflictError:
        # Si le projet existe déjà, on utilise un nom unique
        project_name = f"qa_evaluation_{timestamp}_{os.getpid()}"
        results = client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=chain,
            evaluation=eval_config,
            project_name=project_name,
            metadata={
                "model": "llama3.2",
                "evaluation_date": datetime.now().isoformat(),
                "dataset_id": dataset_id
            }
        )
    
    return project_name, results

def analyze_results(project_name):
    client = Client()
    
    # Récupération des runs avec filtrage
    runs = client.list_runs(
        project_name=project_name,
        filter="has_feedback = true"  # Ne récupère que les runs avec feedback
    )
    
    # Analyse des métriques avec plus de détails
    metrics = {
        "total_runs": len(list(runs)),
        "feedback_scores": [],
        "response_times": [],
        "evaluator_scores": {},
        "metadata_summary": {}
    }
    
    for run in runs:
        if run.feedback:
            for feedback in run.feedback:
                metrics["feedback_scores"].append(feedback.score)
                evaluator_name = feedback.evaluator_name
                if evaluator_name not in metrics["evaluator_scores"]:
                    metrics["evaluator_scores"][evaluator_name] = []
                metrics["evaluator_scores"][evaluator_name].append(feedback.score)
        
        if run.end_time and run.start_time:
            metrics["response_times"].append((run.end_time - run.start_time).total_seconds())
        
        # Analyse des métadonnées
        if run.metadata:
            for key, value in run.metadata.items():
                if key not in metrics["metadata_summary"]:
                    metrics["metadata_summary"][key] = []
                metrics["metadata_summary"][key].append(value)
    
    # Calcul des moyennes et statistiques
    if metrics["feedback_scores"]:
        metrics["avg_score"] = sum(metrics["feedback_scores"]) / len(metrics["feedback_scores"])
    if metrics["response_times"]:
        metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
    
    # Calcul des moyennes par évaluateur
    for evaluator, scores in metrics["evaluator_scores"].items():
        metrics["evaluator_scores"][evaluator] = {
            "average": sum(scores) / len(scores),
            "count": len(scores)
        }
    
    return metrics

if __name__ == "__main__":
    # Exemple d'utilisation
    dataset_name = "evaluation_dataset_prod"  # À remplacer par le nom de votre dataset
    dataset_id = "your_dataset_id"  # À remplacer par l'ID de votre dataset
    
    project_name, results = setup_evaluation(dataset_name, dataset_id)
    metrics = analyze_results(project_name)
    print("Métriques du projet:", metrics) 