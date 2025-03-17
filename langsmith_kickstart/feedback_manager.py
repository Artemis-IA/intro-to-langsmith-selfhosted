import os
from datetime import datetime
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

def process_feedback(run_id, feedback, dataset_id):
    client = Client()
    run = client.read_run(run_id)
    
    # Analyse plus sophistiquée du feedback
    if feedback.score < 0.7:  # Score en dessous de 70%
        # Création d'un exemple avec métadonnées enrichies
        client.create_example(
            inputs=run.inputs,
            outputs=run.outputs,
            dataset_id=dataset_id,
            feedback=[feedback],
            metadata={
                "feedback_score": feedback.score,
                "feedback_type": feedback.evaluator_name,
                "feedback_date": datetime.now().isoformat(),
                "original_run_id": run_id
            }
        )
        return True
    return False

def setup_feedback_loop(dataset_id):
    client = Client()
    
    # Configuration des déclencheurs de feedback avec gestion des erreurs
    def feedback_handler(run_id, feedback):
        try:
            return process_feedback(run_id, feedback, dataset_id)
        except Exception as e:
            print(f"Erreur lors du traitement du feedback pour le run {run_id}: {str(e)}")
            return False
    
    # Configuration des déclencheurs de feedback
    feedback_config = {
        "type": "manual",  # Feedback manuel initial
        "handlers": [feedback_handler],
        "metadata": {
            "setup_date": datetime.now().isoformat(),
            "dataset_id": dataset_id
        }
    }
    
    return feedback_config

def get_feedback_summary(project_name):
    client = Client()
    
    # Récupération des feedbacks avec filtrage
    feedbacks = client.list_feedback(
        project_name=project_name,
        filter="score < 0.7"  # Ne récupère que les feedbacks négatifs
    )
    
    # Analyse des feedbacks avec plus de détails
    summary = {
        "total_feedback": len(feedbacks),
        "average_score": 0,
        "score_distribution": {},
        "evaluator_distribution": {},
        "feedback_trends": {
            "improvement_needed": [],
            "good_performance": []
        }
    }
    
    if feedbacks:
        scores = [f.score for f in feedbacks]
        summary["average_score"] = sum(scores) / len(scores)
        
        # Distribution des scores
        for score in scores:
            bucket = round(score * 10) / 10  # Arrondir à 0.1
            summary["score_distribution"][bucket] = summary["score_distribution"].get(bucket, 0) + 1
        
        # Distribution par évaluateur
        for feedback in feedbacks:
            evaluator = feedback.evaluator_name
            if evaluator not in summary["evaluator_distribution"]:
                summary["evaluator_distribution"][evaluator] = {
                    "count": 0,
                    "average_score": 0,
                    "scores": []
                }
            summary["evaluator_distribution"][evaluator]["count"] += 1
            summary["evaluator_distribution"][evaluator]["scores"].append(feedback.score)
        
        # Calcul des moyennes par évaluateur
        for evaluator in summary["evaluator_distribution"]:
            scores = summary["evaluator_distribution"][evaluator]["scores"]
            summary["evaluator_distribution"][evaluator]["average_score"] = sum(scores) / len(scores)
    
    return summary

def analyze_feedback_trends(project_name, days=7):
    client = Client()
    
    # Récupération des feedbacks sur une période donnée
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    feedbacks = client.list_feedback(
        project_name=project_name,
        filter=f"created_at >= '{start_date.isoformat()}'"
    )
    
    # Analyse des tendances
    trends = {
        "daily_scores": {},
        "evaluator_trends": {},
        "improvement_areas": set()
    }
    
    for feedback in feedbacks:
        date = feedback.created_at.date().isoformat()
        if date not in trends["daily_scores"]:
            trends["daily_scores"][date] = []
        trends["daily_scores"][date].append(feedback.score)
        
        evaluator = feedback.evaluator_name
        if evaluator not in trends["evaluator_trends"]:
            trends["evaluator_trends"][evaluator] = []
        trends["evaluator_trends"][evaluator].append(feedback.score)
        
        if feedback.score < 0.7:
            trends["improvement_areas"].add(evaluator)
    
    # Calcul des moyennes quotidiennes
    for date in trends["daily_scores"]:
        trends["daily_scores"][date] = sum(trends["daily_scores"][date]) / len(trends["daily_scores"][date])
    
    # Calcul des moyennes par évaluateur
    for evaluator in trends["evaluator_trends"]:
        trends["evaluator_trends"][evaluator] = sum(trends["evaluator_trends"][evaluator]) / len(trends["evaluator_trends"][evaluator])
    
    return trends

if __name__ == "__main__":
    # Exemple d'utilisation
    project_name = "qa_evaluation_prod"  # À remplacer par le nom de votre projet
    dataset_id = "your_dataset_id"  # À remplacer par l'ID de votre dataset
    
    feedback_config = setup_feedback_loop(dataset_id)
    summary = get_feedback_summary(project_name)
    trends = analyze_feedback_trends(project_name)
    
    print("Résumé des feedbacks:", summary)
    print("\nTendances des feedbacks:", trends) 