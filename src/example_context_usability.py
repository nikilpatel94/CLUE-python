#%%
from evaluate import calculate_contexts_usability
print("----Simple Question---:")
question1 = "What is the capital of France?"
contexts1 = ["Paris is the capital and most populous city of France.",
             "I am huge fan of MacDonalds.",
            "The capital cities of the world are usually the most populous cities in their respective countries.",
            "Reddit is a better social media platform than every other out there."]
scores1 = calculate_contexts_usability(question=question1,retrieved_contexts=contexts1,generated_output=None,max_output_sequences=1)
print(f"Generated output: {scores1.generated_output}")
print("Context | Usability Score")
print('\n'.join(f"{contexts1[idx]} | {round(scores1.usability_scores[idx],3)}" for idx in range(len(contexts1))))

#%%
print("----Tricky Question---:")
question2 = "Who won Border-Gavaskar trophy in 2023??"
contexts2 = [
"""The Border–Gavaskar Trophy is one of the premier bilateral trophies in Test cricket. Both teams have a reputation of being difficult to beat at home. """,
"""This is borne out by India winning 8 out of 9 series held in India, and Australia winning 4 out of 7 series held in Australia, as of the conclusion of the 2022–23 series. """,
"""The away wins achieved by Australia (2004–05) and India (2018–19 and 2020–21) have earned places in cricket folklore. Both teams have achieved similar number of Test and series wins, and the trophy has changed hands frequently. 
The competitiveness of the series is also reflected in that in both 2000–01 and 2007–08, it was India who ended Australian streaks of 16 consecutive Test wins. """,
"""The 2000–01 series was labelled as the "final frontier" for Australia by their captain Steve Waugh due to the difficulty of winning in India, and was closely fought on both sides. """
]
outputs2 = ["I do not know."]
scores2 = calculate_contexts_usability(question=question2,retrieved_contexts=contexts2,generated_output=outputs2)
print("Context | Usability Score")
print('\n'.join(f"{contexts2[idx]} | {round(scores2.usability_scores[idx],3)}" for idx in range(len(contexts2))))


# %%
