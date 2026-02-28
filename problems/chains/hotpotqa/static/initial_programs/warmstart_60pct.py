def entrypoint():
    return {
        "system_prompt": "You are an expert in multi-hop question answering. At every step, produce only the minimal required output without any extra text. Be precise and concise. For query generation steps, output ONLY the search query string. For evidence steps, output ONLY relevant facts or combined evidence. For final answer, output EXACTLY 'Answer: <answer>'.",
        "steps": [
            # Step 1: First-hop retrieval (frozen tool step)
            {
                "number": 1,
                "title": "Retrieve first-hop passages.",
                "step_type": "tool",
                "step_config": {
                    "tool_name": "retrieve",
                    "input_mapping": {"query": "$outer_context"},
                },
                "dependencies": [],
                "frozen": True,
            },
            # Step 2: Summarize first-hop passages
            {
                "number": 2,
                "title": "Summarize key facts from passages.",
                "step_type": "llm",
                "aim": "Extract key facts from the provided passages relevant to the question.",
                "stage_action": (
                    "First, extract all facts that directly answer the question or describe attributes of the main entities (e.g., nationality, birth year, occupation). "
                    "Second, extract facts that establish relationships between entities required for multi-hop reasoning (e.g., 'X wrote Y' when the question is about Y's author). "
                    "Main entities are those explicitly mentioned in the question or directly answering it. "
                    "Exclude facts about entities that are not connected to these main entities via any direct or indirect relationship. "
                    "List each fact as a short phrase without conclusions, interpretations, or inferences — only include facts explicitly stated in the passages."
                ),
                "reasoning_questions": (
                    "Which facts directly involve the main entities in the question? "
                    "Which facts describe entity attributes or establish critical relationships? "
                    "Exclude facts about unrelated entities."
                ),
                "example_reasoning": (
                    "Question: What is the birth year of the author of 'Pride and Prejudice'?\n"
                    "Passages: [0] Jane Austen | Jane Austen was an English novelist. She wrote 'Pride and Prejudice' in 1813. [1] Pride and Prejudice | This novel was published in 1813.\n"
                    "- Jane Austen: English novelist\n"
                    "- Jane Austen: author of 'Pride and Prejudice'\n"
                    "- 'Pride and Prejudice': published in 1813"
                ),
                "dependencies": [1],
                "frozen": False,
            },
            # Step 3: Generate second-hop query
            {
                "number": 3,
                "title": "Generate second-hop query.",
                "step_type": "llm",
                "aim": "Identify missing information and generate a search query.",
                "stage_action": (
                    "Determine the specific missing fact needed to answer the question. "
                    "Output ONLY the search query as a string of space-separated terms. "
                    "Remember: the output of this step is used directly as the search query. Therefore, output ONLY the query string without any additional text, formatting, or explanations. "
                    "The query must contain the minimal terms necessary to retrieve the missing fact, but include disambiguating context when the entity might be ambiguous: "
                    "  - If the entity name is known and unambiguous, use the entity name and missing fact (e.g., 'Jane Austen birth year'). "
                    "  - If the entity name is ambiguous or unknown, include relevant context from the first-hop summary to disambiguate (e.g., 'English novelist Jane Austen birth year' or 'author of Pride and Prejudice birth year'). "
                    "For relationship-based facts, include the relationship in the query (e.g., 'telephone inventor'). Avoid adding extra terms that are not necessary for disambiguation or fact retrieval (like 'biography')."
                ),
                "reasoning_questions": (
                    "What specific fact is missing to answer the question? "
                    "Is the entity name known from first-hop facts? "
                    "How can we phrase the most precise minimal query?"
                ),
                "example_reasoning": (
                    "Question: What is the birth year of the author of 'Pride and Prejudice'?\n"
                    "First-hop summary: Jane Austen: author of 'Pride and Prejudice'\n"
                    "Jane Austen birth year\n\n"
                    
                    "Question: Who invented the telephone?\n"
                    "First-hop summary: Alexander Graham Bell is credited with inventing the telephone in 1876\n"
                    "telephone inventor\n\n"
                    
                    "Question: What is the current population of Tokyo?\n"
                    "First-hop summary: Tokyo: capital of Japan\n"
                    "Tokyo current population"
                ),
                "dependencies": [2],
                "frozen": False,
            },
            # Step 4: Second-hop retrieval (frozen tool step)
            {
                "number": 4,
                "title": "Retrieve second-hop passages.",
                "step_type": "tool",
                "step_config": {
                    "tool_name": "retrieve",
                    "input_mapping": {"query": "$history[-1]"},
                },
                "dependencies": [3],
                "frozen": True,
            },
            # Step 5: Combine evidence
            {
                "number": 5,
                "title": "Combine evidence",
                "step_type": "llm",
                "aim": "Integrate evidence from both hops using source reliability assessment.",
                "stage_action": (
                    "Combine the first-hop summary and second-hop passages into a unified evidence set. "
                    "When resolving contradictions:\n"
                    "1. Prefer sources that have BOTH:\n"
                    "   a) a title that matches the key entity in the question (e.g., for 'birth year of X', title should be 'X'), AND\n"
                    "   b) a phrase that is highly specific to the fact type (e.g., for birth year: 'born in', 'birth year is').\n"
                    "2. If no source meets both criteria, prefer sources that meet (a) over (b), and if neither, prefer the source with the most detailed description of the fact.\n"
                    "If evidence is consistent, combine facts. Do not invent conflicts."
                ),
                "reasoning_questions": (
                    "What are the key facts from each hop? Are there contradictions? "
                    "Check if sources meet both reliability criteria (entity-relevant title + fact-specific context). "
                    "How do facts complement when consistent?"
                ),
                "example_reasoning": (
                    "Example 1 (birth):\n"
                    "Question: What is the birth year of Jane Austen?\n"
                    "First-hop summary: Jane Austen wrote 'Pride and Prejudice' in 1813.\n"
                    "Second-hop passages: [0] Jane Austen | She was born on December 16, 1775. [1] English novelists | Born in 1770.\n"
                    "Analysis: Contradiction in birth years. Passage [0] has title 'Jane Austen' (key entity) and specific context 'born on'.\n"
                    "Jane Austen was born in 1775.\n\n"
                    
                    "Example 2 (invention):\n"
                    "Question: Who invented the telephone?\n"
                    "First-hop summary: Alexander Graham Bell is credited with inventing the telephone in 1876.\n"
                    "Second-hop passages: [0] Antonio Meucci | Meucci demonstrated a voice-communication device in 1860. [1] Telephone | The patent for the telephone was awarded to Alexander Graham Bell in 1876.\n"
                    "Analysis: Contradiction between Meucci and Bell. Passage [1] has title 'Telephone' (key entity in question) and specific context 'patent for the telephone'.\n"
                    "Alexander Graham Bell invented the telephone.\n\n"
                    
                    "Example 3 (population):\n"
                    "Question: What is the current population of Tokyo?\n"
                    "First-hop summary: Tokyo is the capital of Japan.\n"
                    "Second-hop passages: [0] Tokyo | Population: 13.96 million (2020). [1] Japan | Tokyo population was 13.5 million in 2015.\n"
                    "Analysis: Contradiction in population numbers. Passage [0] has title 'Tokyo' (key entity) and specific context 'Population:'.\n"
                    "Tokyo population is 13.96 million.\n\n"
                    
                    "Example 4 (sports):\n"
                    "Question: Who won the men's 100m at the 2020 Olympics?\n"
                    "First-hop summary: The 2020 Olympics were held in Tokyo.\n"
                    "Second-hop passages: [0] Athletics at the 2020 Summer Olympics | Marcell Jacobs won the men's 100m. [1] Marcell Jacobs | Italian sprinter, Olympic champion in 100m.\n"
                    "Analysis: Both passages agree. Passage [0] has title 'Athletics at the 2020 Summer Olympics' which directly names the event.\n"
                    "Marcell Jacobs won the men's 100m.\n\n"
                    
                    "Example 5 (equal sources):\n"
                    "Question: What is the population of Tokyo in 2020?\n"
                    "First-hop summary: Tokyo is the capital of Japan.\n"
                    "Second-hop passages: [0] Tokyo | Population: 13.96 million (2020). [1] Tokyo Metropolitan Government | Population: 13.9 million (2020).\n"
                    "Analysis: Both passages have titles related to Tokyo (key entity) and provide population for 2020. Passage [0] has more precise number (13.96 vs 13.9).\n"
                    "Tokyo population is 13.96 million."
                ),
                "dependencies": [2, 4],
                "frozen": False,
            },
            # Step 6: Final answer
            {
                "number": 6,
                "title": "Final answer",
                "step_type": "llm",
                "aim": "Answer the question using all gathered evidence.",
                "stage_action": (
                    "Extract the minimal direct answer phrase that directly answers the question. "
                    "- If the question specifies context (e.g., year, location), omit that context in the answer. "
                    "- Include units that are part of the value (e.g., '14.0 million' for population). "
                    "- If the question asks for a year, output ONLY the year (e.g., '2020'). "
                    "Output EXACTLY in format: Answer: <answer>"
                ),
                "reasoning_questions": (
                    "What is the shortest phrase answering the question? "
                    "- For time-specific questions: Is the year the answer or part of context? "
                    "- For value questions: Are units required? "
                    "Verify against evidence and question requirements."
                ),
                "example_reasoning": (
                    "Question: When was the Eiffel Tower completed?\n"
                    "Combined evidence: \n- The Eiffel Tower was completed in 1889.\n"
                    "Reasoning: The question asks for the completion year. The evidence states 1889. "
                    "The minimal answer is '1889' (year only). We omit 'in' and the subject/verb as they're in the question.\n"
                    "Answer: 1889"
                ),
                "dependencies": [2, 5],
                "frozen": False,
            },
        ],
    }