#%%
from evaluate import calculate_model_uncertainty

print("----Simple Question---:")
question1 = "What is the capital of France?"
contexts1 = ["Paris is the capital and most populous city of France.",
             "I am huge fan of MacDonalds.",
            "The capital cities of the world are usually the most populous cities in their respective countries.",
            "Reddit is a better social media platform than every other out there."]
scores1 = calculate_model_uncertainty(question=question1,retrieved_contexts=contexts1)
print(f"Generated Output Sequences: {scores1.output_sequences}")
print("Concept Level Uncertainty Scores")
print('\n'.join(f"{contexts1[idx]} | {round(scores1.uncertainty_scores[idx],3)}" for idx in range(len(contexts1))))

# %%
