"""
Prompt generation utilities for targeted analysis.
"""

from typing import List, Dict, Optional, Tuple
import random
import logging

logger = logging.getLogger(__name__)


class PromptGenerator:
    """
    Generate prompts designed to activate specific expert patterns.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize prompt generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        
    def generate_capability_prompts(self) -> Dict[str, List[str]]:
        """
        Generate prompts targeting different capabilities.
        
        Returns:
            Dictionary mapping capability names to prompt lists
        """
        capability_prompts = {
            "mathematical_reasoning": [
                "Solve this step by step: If 3x + 7 = 22, what is x?",
                "A train travels 120 miles in 2 hours. What is its average speed?",
                "Calculate the area of a circle with radius 5 units.",
                "If the probability of rain is 0.3, what's the probability it won't rain?",
                "Simplify the expression: (2x + 3)(x - 1)",
                "What is the derivative of f(x) = x² + 3x + 2?",
                "Solve the system: x + y = 5, 2x - y = 1",
                "Convert 75% to a decimal and a fraction.",
            ],
            
            "code_generation": [
                "Write a Python function to reverse a string:",
                "Implement a binary search algorithm in Python:",
                "Create a class to represent a binary tree node:",
                "Write a function to find the factorial of a number:",
                "Implement bubble sort in Python:",
                "Create a function to check if a number is prime:",
                "Write code to merge two sorted lists:",
                "Implement a stack data structure using a list:",
            ],
            
            "logical_reasoning": [
                "If all cats are mammals, and Fluffy is a cat, what can we conclude?",
                "Given: All birds can fly. Penguins are birds. What's the logical issue?",
                "If it's raining, then the ground is wet. The ground is wet. Is it raining?",
                "Complete the pattern: 2, 4, 8, 16, ?",
                "If A implies B, and B implies C, what can we say about A and C?",
                "All roses are flowers. Some flowers are red. Are all roses red?",
                "If today is Monday, then tomorrow is Tuesday. Today is Monday. Therefore?",
                "Either it's day or night. It's not day. What can we conclude?",
            ],
            
            "language_understanding": [
                "Explain the meaning of the idiom 'break the ice':",
                "What is the difference between 'affect' and 'effect'?",
                "Identify the subject and predicate in: 'The quick brown fox jumps.'",
                "What does the word 'serendipity' mean?",
                "Correct the grammar: 'Me and him went to the store.'",
                "What is a synonym for 'ubiquitous'?",
                "Explain the metaphor: 'Time is money.'",
                "What is the past tense of 'lie' (to recline)?",
            ],
            
            "factual_knowledge": [
                "What is the capital of Australia?",
                "Who wrote the novel '1984'?",
                "What is the chemical symbol for gold?",
                "In what year did World War II end?",
                "What is the largest planet in our solar system?",
                "Who painted the Mona Lisa?",
                "What is the speed of light in a vacuum?",
                "Which element has the atomic number 1?",
            ],
            
            "creative_writing": [
                "Write a short story beginning with: 'The door creaked open...'",
                "Compose a haiku about autumn leaves:",
                "Create a dialogue between two characters meeting for the first time:",
                "Write a product description for an imaginary gadget:",
                "Describe a sunset using only metaphors:",
                "Write a limerick about a cat:",
                "Create an opening paragraph for a mystery novel:",
                "Write a persuasive paragraph about the importance of reading:",
            ],
        }
        
        return capability_prompts
    
    def generate_domain_specific_prompts(self) -> Dict[str, List[str]]:
        """
        Generate prompts for specific domains.
        
        Returns:
            Dictionary mapping domain names to prompt lists
        """
        domain_prompts = {
            "science": [
                "Explain the process of photosynthesis:",
                "What is Newton's second law of motion?",
                "Describe the structure of DNA:",
                "How does natural selection work?",
                "What causes the greenhouse effect?",
                "Explain the difference between mitosis and meiosis:",
                "What is the periodic table organized by?",
                "Describe the water cycle:",
            ],
            
            "technology": [
                "How does machine learning work?",
                "What is the difference between HTTP and HTTPS?",
                "Explain what cloud computing is:",
                "How do neural networks learn?",
                "What is blockchain technology?",
                "Describe how GPS works:",
                "What is the Internet of Things (IoT)?",
                "How does encryption protect data?",
            ],
            
            "history": [
                "What caused the American Civil War?",
                "Describe the fall of the Roman Empire:",
                "What was the Industrial Revolution?",
                "Who were the key figures in the Renaissance?",
                "What led to World War I?",
                "Explain the significance of the Magna Carta:",
                "What was the Cold War about?",
                "Describe the ancient Egyptian civilization:",
            ],
            
            "literature": [
                "Analyze the theme of love in Romeo and Juliet:",
                "What is the significance of the green light in The Great Gatsby?",
                "Describe the character development in To Kill a Mockingbird:",
                "What are the main themes in 1984?",
                "Explain the symbolism in Lord of the Flies:",
                "What is the narrative structure of Wuthering Heights?",
                "Analyze the use of irony in Pride and Prejudice:",
                "What is the central conflict in Hamlet?",
            ],
        }
        
        return domain_prompts
    
    def generate_complexity_gradient_prompts(self) -> Dict[str, List[str]]:
        """
        Generate prompts with varying complexity levels.
        
        Returns:
            Dictionary mapping complexity levels to prompt lists
        """
        complexity_prompts = {
            "simple": [
                "What is 2 + 2?",
                "Name a color.",
                "What day comes after Monday?",
                "How many legs does a cat have?",
                "What is the opposite of hot?",
            ],
            
            "medium": [
                "Explain how to make a peanut butter sandwich:",
                "What are the primary colors?",
                "Describe the difference between weather and climate:",
                "How do you calculate the area of a rectangle?",
                "What is the water cycle?",
            ],
            
            "complex": [
                "Analyze the economic implications of artificial intelligence on employment:",
                "Discuss the ethical considerations in genetic engineering:",
                "Explain the relationship between quantum mechanics and general relativity:",
                "Evaluate the impact of social media on democratic processes:",
                "Describe the challenges of sustainable urban development:",
            ],
            
            "expert": [
                "Derive the Schrödinger equation from first principles:",
                "Analyze the computational complexity of the traveling salesman problem:",
                "Discuss the implications of Gödel's incompleteness theorems:",
                "Explain the mechanism of CRISPR-Cas9 gene editing:",
                "Evaluate the convergence properties of stochastic gradient descent:",
            ],
        }
        
        return complexity_prompts
    
    def generate_adversarial_prompts(self) -> List[str]:
        """
        Generate prompts designed to test edge cases and potential failure modes.
        
        Returns:
            List of adversarial prompts
        """
        adversarial_prompts = [
            # Ambiguous prompts
            "The bank was steep.",  # Financial institution vs. river bank
            "I saw her duck.",  # Verb vs. noun
            "Time flies like an arrow.",  # Multiple interpretations
            
            # Contradictory information
            "The colorless green ideas sleep furiously.",
            "This statement is false.",
            "Can an omnipotent being create a stone so heavy they cannot lift it?",
            
            # Edge cases
            "What is the square root of -1?",
            "Divide by zero.",
            "What happens when an unstoppable force meets an immovable object?",
            
            # Context switching
            "In Python, how do you... wait, actually, let's talk about Java instead.",
            "Solve this math problem: Actually, can you write a poem instead?",
            
            # Incomplete information
            "Based on the data provided above, what can we conclude?",  # No data provided
            "As mentioned earlier, the solution is...",  # Nothing mentioned earlier
        ]
        
        return adversarial_prompts
    
    def generate_expert_activation_prompts(
        self,
        target_experts: List[int],
        layer_idx: int
    ) -> List[str]:
        """
        Generate prompts designed to activate specific experts.
        This is experimental and would need to be refined based on actual routing patterns.
        
        Args:
            target_experts: List of expert indices to target
            layer_idx: Layer index
            
        Returns:
            List of prompts designed to activate target experts
        """
        # This is a placeholder - in practice, you'd need to analyze
        # which types of prompts activate which experts
        
        expert_type_mapping = {
            0: "mathematical",
            1: "linguistic", 
            2: "logical",
            3: "factual",
            4: "creative",
            5: "technical",
            6: "analytical",
            7: "general",
        }
        
        prompts = []
        capability_prompts = self.generate_capability_prompts()
        
        for expert_idx in target_experts:
            expert_type = expert_type_mapping.get(expert_idx % 8, "general")
            
            if expert_type == "mathematical":
                prompts.extend(capability_prompts["mathematical_reasoning"][:2])
            elif expert_type == "linguistic":
                prompts.extend(capability_prompts["language_understanding"][:2])
            elif expert_type == "logical":
                prompts.extend(capability_prompts["logical_reasoning"][:2])
            elif expert_type == "creative":
                prompts.extend(capability_prompts["creative_writing"][:2])
            else:
                prompts.extend(capability_prompts["factual_knowledge"][:2])
        
        return prompts
    
    def create_balanced_prompt_set(
        self,
        samples_per_category: int = 50
    ) -> Tuple[List[str], List[str]]:
        """
        Create a balanced set of prompts across different categories.
        
        Args:
            samples_per_category: Number of samples per category
            
        Returns:
            Tuple of (prompts, labels)
        """
        all_prompts = []
        all_labels = []
        
        # Add capability prompts
        capability_prompts = self.generate_capability_prompts()
        for capability, prompts in capability_prompts.items():
            selected_prompts = random.choices(prompts, k=min(samples_per_category, len(prompts)))
            all_prompts.extend(selected_prompts)
            all_labels.extend([capability] * len(selected_prompts))
        
        # Add domain prompts
        domain_prompts = self.generate_domain_specific_prompts()
        for domain, prompts in domain_prompts.items():
            selected_prompts = random.choices(prompts, k=min(samples_per_category, len(prompts)))
            all_prompts.extend(selected_prompts)
            all_labels.extend([domain] * len(selected_prompts))
        
        # Add complexity prompts
        complexity_prompts = self.generate_complexity_gradient_prompts()
        for complexity, prompts in complexity_prompts.items():
            selected_prompts = random.choices(prompts, k=min(samples_per_category, len(prompts)))
            all_prompts.extend(selected_prompts)
            all_labels.extend([f"complexity_{complexity}"] * len(selected_prompts))
        
        # Shuffle the combined set
        combined = list(zip(all_prompts, all_labels))
        random.shuffle(combined)
        all_prompts, all_labels = zip(*combined)
        
        logger.info(f"Created balanced prompt set with {len(all_prompts)} prompts")
        return list(all_prompts), list(all_labels)

