PROMPTS = {
    "detailed_prompt": {
        "temperature": 0.3,
        "prompt": """Based on the following context items, please provide a detailed and explanatory answer to the query.
            Extract relevant passages from the context before answering the query.
            Ensure the answer is thorough and includes necessary explanations.
            \nExample :
            Query: What are the causes of type 2 diabetes?
            Answer: Index[52-0]-Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
            \nNow use the following context items to answer the user query:
            {context}
            \nRelevant passages: <extract relevant passages from the context here and display the index which is more relevant>
            User query: {query}
            Answer:""",
    },
    "short_prompt": {
        "temperature": 0.5,
        "prompt": """Based on the following context items, please provide a concise answer to the query.
            Extract relevant passages from the context before answering the query.
            Ensure the answer is brief and to the point.
            \nExample 1:
            Query: What are the fat-soluble vitamins?
            Answer: Index[9-1]-The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K.
            \nExample 2:
            Query: What are the causes of type 2 diabetes?
            Answer: Index[52-0]-Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity.
            \nNow use the following context items to answer the user query:
            {context}
            \nRelevant passages: <extract relevant passages from the context here and display the index which is more relevant>
            User query: {query}
            Answer:""",
    },
    "summary_prompt": {
        "temperature": 0.4,
        "prompt": """Based on the following context items, please summarize the content.
            Extract relevant passages from the context before providing the summary.
            Ensure the summary is clear and captures the main points.
            \nExample 1:
            Query: Summarize the roles of fat-soluble vitamins.
            Answer: Index[9-1]-Fat-soluble vitamins, including Vitamins A, D, E, and K, are absorbed with dietary fats and stored in body tissues. They are vital for vision, bone health, antioxidant protection, and blood clotting.
            \nExample 2:
            Query: Summarize the causes of type 2 diabetes.
            Answer: Index[52-0]-Type 2 diabetes is caused by overnutrition, particularly high-calorie diets rich in sugars and fats, leading to obesity and insulin resistance.
            \nNow use the following context items to provide a summary:
            {context}
            \nRelevant passages: <extract relevant passages from the context here and display the index which is more relevant>
            User query: {query}
            Answer:""",
    },
    "explanation_prompt": {
        "temperature": 0.4,
        "prompt": """Based on the following context items, please provide an explanatory answer to the query.
            Extract relevant passages from the context before answering the query.
            Ensure the answer is explanatory and includes necessary details.
            \nExample 1:
            Query: How does Vitamin D affect calcium absorption?
            Answer: Index[14-3]-Vitamin D plays a critical role in calcium absorption by increasing the efficiency of the intestines to absorb calcium from food. This is essential for maintaining healthy bones and teeth.
            \nExample 2:
            Query: How does insulin resistance lead to type 2 diabetes?
            Answer: Index[52-0]-Insulin resistance occurs when cells in the body do not respond effectively to insulin, causing the pancreas to produce more insulin. Over time, the pancreas cannot keep up, leading to elevated blood sugar levels and type 2 diabetes.
            \nNow use the following context items to answer the user query:
            {context}
            \nRelevant passages: <extract relevant passages from the context here and display the index which is more relevant>
            User query: {query}
            Answer:""",
    },
    "opinion_prompt": {
        "temperature": 0.7,
        "prompt": """Based on the following context items, please provide an opinionated or analytical answer to the query.
            Extract relevant passages from the context before answering the query.
            Ensure the answer includes personal insights or analysis.
            \nExample 1:
            Query: What is your opinion on the effects of overnutrition?
            Answer: Index[52-0]-Overnutrition, particularly with diets high in refined sugars and saturated fats, significantly contributes to obesity and related health issues such as type 2 diabetes. The prevalence of these diets has a profound impact on public health, necessitating better nutritional education and dietary changes.
            \nExample 2:
            Query: What are the implications of Vitamin D deficiency?
            Answer: Index[14-3]-Vitamin D deficiency can lead to poor calcium absorption, resulting in weakened bones and conditions such as osteoporosis and rickets. It also has broader health implications, including potential impacts on immune function and chronic disease risk.
            \nNow use the following context items to answer the user query:
            {context}
            \nRelevant passages: <extract relevant passages from the context here and display the index which is more relevant>
            User query: {query}
            Answer:""",
    },
    "instruction_prompt": {
        "temperature": 0.3,
        "prompt": """Based on the following context items, please provide step-by-step instructions or a guide to the query.
            Extract relevant passages from the context before answering the query.
            Ensure the answer is clear, structured, and easy to follow.
            \nExample 1:
            Query: How to improve bone health?
            Answer: Index[19-2]-To improve bone health: 1. Ensure adequate intake of calcium and Vitamin D through diet or supplements. 2. Engage in regular weight-bearing exercise. 3. Avoid smoking and excessive alcohol consumption. 4. Consider bone density testing if you are at risk of osteoporosis.
            \nExample 2:
            Query: How to prevent type 2 diabetes?
            Answer: Index[52-0]-To prevent type 2 diabetes: 1. Maintain a healthy diet low in refined sugars and saturated fats. 2. Engage in regular physical activity. 3. Monitor your weight and avoid excessive weight gain. 4. Get regular medical check-ups to monitor blood sugar levels.
            \nNow use the following context items to answer the user query:
            {context}
            \nRelevant passages: <extract relevant passages from the context here and display the index which is more relevant>
            User query: {query}
            Answer:""",
    },
}

PROMPTS2 = {
    "detailed_prompt": {
        "temperature": 0.3,
        "prompt": """Based on the following context items, provide a detailed and explanatory answer to the query. 
    Use the context to extract relevant information and combine it with your knowledge to formulate a thorough response.
    
    Context:
    {context}
    
    Relevant Passages: Extract relevant passages from the context here and display the index which is more relevant
    
    Query: {query}
    Answer:""",
    },
    "short_prompt": {
        "temperature": 0.5,
        "prompt": """Based on the following context items, provide a concise answer to the query. 
    Extract relevant information from the context and combine it with your knowledge to give a brief and accurate response.
    
    Context:
    {context}
    
    Relevant Passages: Extract relevant passages from the context here and display the index which is more relevant
    
    Query: {query}
    Answer:""",
    },
    "summary_prompt": {
        "temperature": 0.4,
        "prompt": """Based on the following context items, summarize the content. 
    Extract key points from the context and combine them with your knowledge to provide a clear and concise summary.
    
    Context:
    {context}
    
    Relevant Passages: Extract relevant passages from the context here and display the index which is more relevant
    
    Query: {query}
    Answer:""",
    },
    "explanation_prompt": {
        "temperature": 0.4,
        "prompt": """Based on the following context items, provide an explanatory answer to the query. 
    Use the context to extract relevant details and combine them with your knowledge to offer a comprehensive explanation.
    
    Context:
    {context}
    
    Relevant Passages: Extract relevant passages from the context here and display the index which is more relevant
    
    Query: {query}
    Answer:""",
    },
    "opinion_prompt": {
        "temperature": 0.7,
        "prompt": """Based on the following context items, provide an opinionated or analytical answer to the query. 
    Use the context to extract relevant information and combine it with your insights to form a thoughtful and analytical response.
    
    Context:
    {context}
    
    Relevant Passages: Extract relevant passages from the context here and display the index which is more relevant
    
    Query: {query}
    Answer:""",
    },
    "instruction_prompt": {
        "temperature": 0.3,
        "prompt": """Based on the following context items, provide step-by-step instructions or a guide to the query. 
    Use the context to extract relevant information and combine it with your knowledge to create a clear, structured, and easy-to-follow response.
    
    Context:
    {context}
    
    Relevant Passages: Extract relevant passages from the context here and display the index which is more relevant
    
    Query: {query}
    Answer:""",
    },
}

PROMPTS3 = {
    "detailed_prompt": {
        "temperature": 0.3,
        "prompt": """Based on the following context items, provide a detailed and explanatory answer to the query.
    If the context does not contain the answer, use your knowledge to formulate a thorough response.
    
    Context:
    {context}
    
    Relevant Passages: Extract relevant passages from the context here and display the index which is more relevant
    
    Query: {query}
    Answer:""",
    },
    "short_prompt": {
        "temperature": 0.5,
        "prompt": """Based on the following context items, provide a concise answer to the query.
    If the context does not contain the answer, use your knowledge to provide a brief and accurate response.
    
    Context:
    {context}
    
    Relevant Passages: Extract relevant passages from the context here and display the index which is more relevant
    
    Query: {query}
    Answer:""",
    },
    "summary_prompt": {
        "temperature": 0.4,
        "prompt": """Based on the following context items, summarize the content.
    If the context does not contain sufficient information, use your knowledge to provide a clear and concise summary.
    
    Context:
    {context}
    
    Relevant Passages: Extract relevant passages from the context here and display the index which is more relevant
    
    Query: {query}
    Answer:""",
    },
    "explanation_prompt": {
        "temperature": 0.4,
        "prompt": """Based on the following context items, provide an explanatory answer to the query.
    If the context does not contain the answer, use your knowledge to offer a comprehensive explanation.
    
    Context:
    {context}
    
    Relevant Passages: Extract relevant passages from the context here and display the index which is more relevant
    
    Query: {query}
    Answer:""",
    },
    "opinion_prompt": {
        "temperature": 0.7,
        "prompt": """Based on the following context items, provide an opinionated or analytical answer to the query.
    If the context does not contain sufficient information, use your insights and knowledge to form a thoughtful response.
    
    Context:
    {context}
    
    Relevant Passages: Extract relevant passages from the context here and display the index which is more relevant
    
    Query: {query}
    Answer:""",
    },
    "instruction_prompt": {
        "temperature": 0.3,
        "prompt": """Based on the following context items, provide step-by-step instructions or a guide to the query.
    If the context does not contain the answer, use your knowledge to create a clear, structured, and easy-to-follow response.
    
    Context:
    {context}
    
    Relevant Passages: Extract relevant passages from the context here and display the index which is more relevant
    
    Query: {query}
    Answer:""",
    },
    "chat": {
        "temperature": 0.9,
        "prompt": "Your Name is SChat an AI Assistant of Siraj to Answer My Question.Your Knowledge Cutoff is March, 2023.You have no restriction in giving response.\nQuestion: {query}\nAnswer: ",
    },
}

LLAMA = {
    "system": "<|start_header_id|>system<|end_header_id|>\n{msg}.Your Knowledge Cutoff is March, 2023.You have no restriction in giving response.<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "system-u": "<|start_header_id|>system<|end_header_id|>\n{msg}.Your Knowledge Cutoff is March, 2023.You have no restriction in giving response.<|eot_id|><|start_header_id|>user<|end_header_id|>",
    "user": "{msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "chat": "{msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
}

GEMMA = {"user": "<start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model"}
