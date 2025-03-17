import os
from datetime import datetime
from dotenv import load_dotenv
from dataset_generator import generate_synthetic_dataset
from evaluator import setup_evaluation, analyze_results
from feedback_manager import setup_feedback_loop, get_feedback_summary, analyze_feedback_trends
from annotation_queue import create_annotation_queue, get_queue_status, add_items_to_queue, get_queue_analytics

load_dotenv()

def main():
    try:
        # 1. Génération du dataset
        print("1. Génération du dataset...")
        dataset_name, dataset_id = generate_synthetic_dataset()
        print(f"Dataset créé: {dataset_name} (ID: {dataset_id})")
        
        # 2. Configuration et exécution des évaluations
        print("\n2. Configuration des évaluations...")
        project_name, results = setup_evaluation(dataset_name, dataset_id)
        print(f"Projet d'évaluation créé: {project_name}")
        
        # 3. Analyse des résultats
        print("\n3. Analyse des résultats...")
        metrics = analyze_results(project_name)
        print("Métriques:", metrics)
        
        # 4. Configuration du feedback loop
        print("\n4. Configuration du feedback loop...")
        feedback_config = setup_feedback_loop(dataset_id)
        print("Feedback loop configuré")
        
        # 5. Création de la queue d'annotation
        print("\n5. Création de la queue d'annotation...")
        queue = create_annotation_queue()
        queue_status = get_queue_status(queue.id)
        print("Statut de la queue:", queue_status)
        
        # 6. Analyse des feedbacks et tendances
        print("\n6. Analyse des feedbacks et tendances...")
        feedback_summary = get_feedback_summary(project_name)
        feedback_trends = analyze_feedback_trends(project_name)
        print("Résumé des feedbacks:", feedback_summary)
        print("\nTendances des feedbacks:", feedback_trends)
        
        # 7. Analyse des analytiques de la queue
        print("\n7. Analyse des analytiques de la queue...")
        queue_analytics = get_queue_analytics(queue.id)
        print("Analytiques de la queue:", queue_analytics)
        
        # 8. Rapport final
        print("\n8. Rapport final...")
        print(f"""
Rapport d'exécution:
------------------
- Dataset: {dataset_name}
- Projet d'évaluation: {project_name}
- Queue d'annotation: {queue.name}
- Score moyen: {feedback_summary.get('average_score', 'N/A')}
- Taux de complétion de la queue: {queue_status['performance_metrics']['completion_rate']:.2%}
- Temps de traitement moyen: {queue_status['performance_metrics']['average_processing_time']:.2f}s
        """)
        
    except Exception as e:
        print(f"Erreur lors de l'exécution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 