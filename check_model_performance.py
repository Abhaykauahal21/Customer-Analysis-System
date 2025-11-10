import json

# Load the visualization data
with open('ml_pipeline/results/visualization_data.json', 'r') as f:
    data = json.load(f)

models = data.get('models', [])

print("="*80)
print("MODEL PREDICTION PERFORMANCE METRICS")
print("="*80)
print()

for model in models:
    name = model.get('name', 'Unknown')
    accuracy = model.get('accuracy', 0) * 100
    precision = model.get('precision', 0) * 100
    recall = model.get('recall', 0) * 100
    f1_score = model.get('f1_score', 0) * 100
    roc_auc = model.get('roc_auc', 0) * 100 if model.get('roc_auc') else None
    
    print(f"📊 {name}")
    print(f"   Accuracy:     {accuracy:.2f}%")
    print(f"   Precision:    {precision:.2f}%")
    print(f"   Recall:       {recall:.2f}%")
    print(f"   F1-Score:     {f1_score:.2f}%")
    if roc_auc:
        print(f"   ROC-AUC:      {roc_auc:.2f}%")
    print()

# Find best model
if models:
    best_model = max(models, key=lambda x: x.get('f1_score', 0))
    print("="*80)
    print(f"🏆 BEST MODEL: {best_model.get('name')}")
    print(f"   Selected based on highest F1-Score: {best_model.get('f1_score', 0)*100:.2f}%")
    print("="*80)

