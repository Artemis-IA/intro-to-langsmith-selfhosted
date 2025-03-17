import os
from datetime import datetime
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

def create_annotation_queue():
    client = Client()
    
    # Création d'une queue d'annotation avec métadonnées
    queue_name = f"qa_annotation_queue_{datetime.now().strftime('%Y%m%d')}"
    annotation_queue = client.create_queue(
        name=queue_name,
        description="Queue pour l'annotation manuelle des réponses",
        metadata={
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "purpose": "QA evaluation"
        }
    )
    
    # Configuration des critères d'annotation avec plus de détails
    annotation_criteria = [
        {
            "name": "accuracy",
            "type": "float",
            "description": "Précision de la réponse (0-1)",
            "metadata": {
                "importance": "high",
                "threshold": 0.8
            }
        },
        {
            "name": "completeness",
            "type": "float",
            "description": "Complétude de la réponse (0-1)",
            "metadata": {
                "importance": "medium",
                "threshold": 0.7
            }
        },
        {
            "name": "clarity",
            "type": "float",
            "description": "Clarté de la réponse (0-1)",
            "metadata": {
                "importance": "medium",
                "threshold": 0.7
            }
        },
        {
            "name": "relevance",
            "type": "float",
            "description": "Pertinence de la réponse (0-1)",
            "metadata": {
                "importance": "high",
                "threshold": 0.8
            }
        }
    ]
    
    # Ajout des critères à la queue avec gestion des erreurs
    for criterion in annotation_criteria:
        try:
            client.create_queue_criterion(
                queue_id=annotation_queue.id,
                name=criterion["name"],
                value_type=criterion["type"],
                description=criterion["description"],
                metadata=criterion.get("metadata", {})
            )
        except Exception as e:
            print(f"Erreur lors de la création du critère {criterion['name']}: {str(e)}")
    
    return annotation_queue

def get_queue_status(queue_id):
    client = Client()
    
    # Récupération des statistiques de la queue avec plus de détails
    queue = client.read_queue(queue_id)
    items = client.list_queue_items(queue_id=queue_id)
    
    status = {
        "queue_name": queue.name,
        "total_items": len(list(items)),
        "pending_items": len([item for item in items if item.status == "pending"]),
        "completed_items": len([item for item in items if item.status == "completed"]),
        "failed_items": len([item for item in items if item.status == "failed"]),
        "criteria": [criterion.name for criterion in queue.criteria],
        "metadata": queue.metadata,
        "performance_metrics": {
            "completion_rate": 0,
            "average_processing_time": 0
        }
    }
    
    # Calcul des métriques de performance
    completed_items = [item for item in items if item.status == "completed"]
    if completed_items:
        status["performance_metrics"]["completion_rate"] = len(completed_items) / status["total_items"]
        
        # Calcul du temps de traitement moyen
        processing_times = []
        for item in completed_items:
            if item.completed_at and item.created_at:
                processing_time = (item.completed_at - item.created_at).total_seconds()
                processing_times.append(processing_time)
        
        if processing_times:
            status["performance_metrics"]["average_processing_time"] = sum(processing_times) / len(processing_times)
    
    return status

def add_items_to_queue(queue_id, run_ids, priority=1):
    client = Client()
    
    # Ajout des runs à la queue d'annotation avec gestion des erreurs
    added_items = []
    failed_items = []
    
    for run_id in run_ids:
        try:
            item = client.create_queue_item(
                queue_id=queue_id,
                run_id=run_id,
                priority=priority,
                metadata={
                    "added_at": datetime.now().isoformat(),
                    "priority": priority
                }
            )
            added_items.append(item.id)
        except Exception as e:
            failed_items.append({
                "run_id": run_id,
                "error": str(e)
            })
    
    return {
        "successful": added_items,
        "failed": failed_items
    }

def get_queue_analytics(queue_id, days=7):
    client = Client()
    
    # Récupération des items sur une période donnée
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    items = client.list_queue_items(
        queue_id=queue_id,
        filter=f"created_at >= '{start_date.isoformat()}'"
    )
    
    # Analyse des données
    analytics = {
        "daily_stats": {},
        "criteria_stats": {},
        "annotator_stats": {},
        "completion_trends": []
    }
    
    for item in items:
        # Statistiques quotidiennes
        date = item.created_at.date().isoformat()
        if date not in analytics["daily_stats"]:
            analytics["daily_stats"][date] = {
                "total": 0,
                "completed": 0,
                "pending": 0,
                "failed": 0
            }
        analytics["daily_stats"][date]["total"] += 1
        analytics["daily_stats"][date][item.status] += 1
        
        # Statistiques par critère
        if item.feedback:
            for feedback in item.feedback:
                criterion = feedback.evaluator_name
                if criterion not in analytics["criteria_stats"]:
                    analytics["criteria_stats"][criterion] = {
                        "total": 0,
                        "average_score": 0,
                        "scores": []
                    }
                analytics["criteria_stats"][criterion]["total"] += 1
                analytics["criteria_stats"][criterion]["scores"].append(feedback.score)
        
        # Statistiques par annotateur
        if item.completed_by:
            annotator = item.completed_by
            if annotator not in analytics["annotator_stats"]:
                analytics["annotator_stats"][annotator] = {
                    "total": 0,
                    "average_time": 0,
                    "completion_times": []
                }
            analytics["annotator_stats"][annotator]["total"] += 1
            if item.completed_at and item.created_at:
                completion_time = (item.completed_at - item.created_at).total_seconds()
                analytics["annotator_stats"][annotator]["completion_times"].append(completion_time)
    
    # Calcul des moyennes
    for criterion in analytics["criteria_stats"]:
        scores = analytics["criteria_stats"][criterion]["scores"]
        if scores:
            analytics["criteria_stats"][criterion]["average_score"] = sum(scores) / len(scores)
    
    for annotator in analytics["annotator_stats"]:
        times = analytics["annotator_stats"][annotator]["completion_times"]
        if times:
            analytics["annotator_stats"][annotator]["average_time"] = sum(times) / len(times)
    
    return analytics

if __name__ == "__main__":
    # Exemple d'utilisation
    queue = create_annotation_queue()
    status = get_queue_status(queue.id)
    analytics = get_queue_analytics(queue.id)
    
    print("Statut de la queue:", status)
    print("\nAnalytiques de la queue:", analytics) 