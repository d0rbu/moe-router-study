#!/usr/bin/env python
"""
Simple demo of k-means autointerpretability concepts.

This script demonstrates the key ideas without requiring external dependencies.
"""

import json
from pathlib import Path
import tempfile


def demo_explanation_structure():
    """Show the structure of a centroid explanation."""
    print("\n" + "=" * 80)
    print("DEMO: Centroid Explanation Structure")
    print("=" * 80 + "\n")

    explanation = {
        "centroid_id": 42,
        "explanation": "This feature activates on punctuation marks and sentence boundaries",
        "confidence": 0.85,
        "top_k_examples": [
            "The cat sat on the mat.",
            "Hello, world!",
            "Is this working?",
            "She said, \"Hello there!\"",
            "What time is it?",
        ],
        "timestamp": "2024-01-01T12:00:00",
        "metadata": {
            "layer_idx": 6,
            "activation_type": "layer_output",
            "num_validation_examples": 10000,
            "generation_method": "llm",
        },
    }

    print("Example Centroid Explanation:")
    print(json.dumps(explanation, indent=2))


def demo_cache_operations():
    """Demonstrate cache operations."""
    print("\n" + "=" * 80)
    print("DEMO: Cache Operations")
    print("=" * 80 + "\n")

    # Simulate cache storage
    with tempfile.TemporaryDirectory() as cache_dir:
        cache_dir = Path(cache_dir)

        print(f"Cache directory: {cache_dir}\n")

        # Store some explanations
        explanations = {
            0: {
                "centroid_id": 0,
                "explanation": "Punctuation and sentence boundaries",
                "confidence": 0.95,
            },
            1: {
                "centroid_id": 1,
                "explanation": "Prepositions indicating location or direction",
                "confidence": 0.78,
            },
            2: {
                "centroid_id": 2,
                "explanation": "Determiners and articles (the, a, an)",
                "confidence": 0.88,
            },
        }

        print("Storing explanations:")
        for cid, explanation in explanations.items():
            cache_file = cache_dir / f"centroid_{cid}.json"
            with open(cache_file, "w") as f:
                json.dump(explanation, f, indent=2)
            print(f"  ✓ Saved centroid_{cid}.json")

        print("\nCache contents:")
        for cache_file in sorted(cache_dir.glob("*.json")):
            print(f"  - {cache_file.name}")

        print("\nRetrieving from cache:")
        cache_file = cache_dir / "centroid_1.json"
        with open(cache_file) as f:
            loaded = json.load(f)
            print(f"  Centroid {loaded['centroid_id']}: {loaded['explanation']}")
            print(f"  Confidence: {loaded['confidence']:.2f}")


def demo_validation_examples():
    """Show the structure of validation examples."""
    print("\n" + "=" * 80)
    print("DEMO: Validation Examples for Top-K Matching")
    print("=" * 80 + "\n")

    examples = [
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "tokens": [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13],
            "token_strings": ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."],
            "activation": 0.95,
            "token_position": 4,  # "jumps"
        },
        {
            "text": "She sells seashells by the seashore.",
            "tokens": [3347, 16417, 37127, 71, 416, 262, 40256, 13],
            "token_strings": ["She", " sells", " seash", "ells", " by", " the", " seashore", "."],
            "activation": 0.87,
            "token_position": 1,  # "sells"
        },
        {
            "text": "Birds fly south for the winter.",
            "tokens": [33, 343, 82, 6574, 5366, 329, 262, 7374, 13],
            "token_strings": ["B", "ir", "ds", " fly", " south", " for", " the", " winter", "."],
            "activation": 0.82,
            "token_position": 3,  # "fly"
        },
    ]

    print("Top-3 validation examples for Centroid 5 (Motion Verbs):\n")
    for i, ex in enumerate(examples, 1):
        activated_token = ex["token_strings"][ex["token_position"]]
        print(f"Example {i} (activation={ex['activation']:.2f}):")
        print(f"  Text: {ex['text']}")
        print(f"  Activated token: '{activated_token}' at position {ex['token_position']}")
        print()


def demo_sentence_analysis_pipeline():
    """Demonstrate the full analysis pipeline."""
    print("\n" + "=" * 80)
    print("DEMO: Full Sentence Analysis Pipeline")
    print("=" * 80 + "\n")

    sentence = "The quick brown fox jumps over the lazy dog."
    print(f"Input sentence: {sentence}\n")

    print("Step 1: Tokenize the sentence")
    tokens = ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."]
    print(f"  Tokens: {tokens}\n")

    print("Step 2: Run through model and extract activations")
    print("  (Running forward pass through layer 6...)")
    print("  (Extracted activations shape: [10 tokens, 768 dimensions])\n")

    print("Step 3: Assign each token to nearest k-means centroid")
    # Simulated centroid assignments
    assignments = [
        {"token": "The", "centroid": 2, "distance": 0.23},
        {"token": " quick", "centroid": 7, "distance": 0.31},
        {"token": " brown", "centroid": 8, "distance": 0.28},
        {"token": " fox", "centroid": 3, "distance": 0.19},
        {"token": " jumps", "centroid": 5, "distance": 0.22},
        {"token": " over", "centroid": 1, "distance": 0.25},
        {"token": " the", "centroid": 2, "distance": 0.24},
        {"token": " lazy", "centroid": 7, "distance": 0.33},
        {"token": " dog", "centroid": 3, "distance": 0.18},
        {"token": ".", "centroid": 0, "distance": 0.15},
    ]

    for i, assignment in enumerate(assignments):
        print(f"  Token {i}: '{assignment['token']:10s}' → Centroid {assignment['centroid']} (dist={assignment['distance']:.2f})")

    print("\nStep 4: Get or generate explanations for unique centroids")
    unique_centroids = {0, 1, 2, 3, 5, 7, 8}
    print(f"  Unique centroids used: {sorted(unique_centroids)}")
    print("  Checking cache...")

    # Simulate cache hits and misses
    cache_status = {
        0: "✓ cache hit",
        1: "✗ cache miss - generating explanation",
        2: "✓ cache hit",
        3: "✓ cache hit",
        5: "✗ cache miss - generating explanation",
        7: "✓ cache hit",
        8: "✗ cache miss - generating explanation",
    }

    for cid in sorted(unique_centroids):
        print(f"  Centroid {cid}: {cache_status[cid]}")

    print("\nStep 5: Build token-level analysis results")
    results = [
        {
            "position": 0,
            "token": "The",
            "centroid": 2,
            "explanation": "Determiners and articles",
            "confidence": 0.88,
            "activation": 1.234,
        },
        {
            "position": 1,
            "token": " quick",
            "centroid": 7,
            "explanation": "Adjectives describing speed or intensity",
            "confidence": 0.75,
            "activation": 0.987,
        },
        {
            "position": 2,
            "token": " brown",
            "centroid": 8,
            "explanation": "Color descriptors",
            "confidence": 0.82,
            "activation": 1.045,
        },
        {
            "position": 3,
            "token": " fox",
            "centroid": 3,
            "explanation": "Animal names",
            "confidence": 0.90,
            "activation": 1.567,
        },
        {
            "position": 4,
            "token": " jumps",
            "centroid": 5,
            "explanation": "Action verbs indicating motion",
            "confidence": 0.85,
            "activation": 1.423,
        },
        {
            "position": 5,
            "token": " over",
            "centroid": 1,
            "explanation": "Prepositions of location/direction",
            "confidence": 0.78,
            "activation": 0.876,
        },
        {
            "position": 6,
            "token": " the",
            "centroid": 2,
            "explanation": "Determiners and articles",
            "confidence": 0.88,
            "activation": 1.198,
        },
        {
            "position": 7,
            "token": " lazy",
            "centroid": 7,
            "explanation": "Adjectives describing speed or intensity",
            "confidence": 0.75,
            "activation": 0.934,
        },
        {
            "position": 8,
            "token": " dog",
            "centroid": 3,
            "explanation": "Animal names",
            "confidence": 0.90,
            "activation": 1.621,
        },
        {
            "position": 9,
            "token": ".",
            "centroid": 0,
            "explanation": "Punctuation and sentence boundaries",
            "confidence": 0.95,
            "activation": 1.789,
        },
    ]

    print("\n" + "-" * 80)
    print(f"{'Pos':<4} {'Token':<12} {'C-ID':<5} {'Conf':<6} {'Act':<7} {'Explanation'}")
    print("-" * 80)
    for result in results:
        print(
            f"{result['position']:<4} "
            f"{result['token']:<12} "
            f"{result['centroid']:<5} "
            f"{result['confidence']:<6.2f} "
            f"{result['activation']:<7.3f} "
            f"{result['explanation']}"
        )

    print("\nStep 6: Generate meta-interpretation")
    print("  (Feeding token analyses to meta-model...)\n")

    meta_interpretation = """
Meta-Interpretation:

The model processes this sentence by activating distinct feature patterns for different
linguistic categories:

1. SYNTACTIC STRUCTURE: The sentence is clearly bounded by punctuation features 
   (centroid 0, 0.95 confidence). Determiners "the" consistently activate centroid 2
   (0.88 confidence), showing stable article recognition.

2. SEMANTIC CATEGORIES: The model distinguishes:
   - Animal nouns ("fox", "dog") → centroid 3 (0.90 confidence, high activation)
   - Motion verbs ("jumps") → centroid 5 (0.85 confidence)
   - Color adjectives ("brown") → centroid 8 (0.82 confidence)
   - Descriptive adjectives ("quick", "lazy") → centroid 7 (0.75 confidence)

3. RELATIONAL ELEMENTS: Prepositions like "over" activate centroid 1, indicating
   spatial/directional relationships between entities.

This reveals hierarchical processing: the model first identifies syntactic roles
(articles, prepositions), then semantic categories (animals, colors, actions).
The consistent activation patterns across similar word types demonstrate organized
internal representations that capture both grammatical structure and meaning.
    """
    print(meta_interpretation.strip())


def demo_use_cases():
    """Show example use cases for the system."""
    print("\n" + "=" * 80)
    print("DEMO: Example Use Cases")
    print("=" * 80 + "\n")

    use_cases = [
        {
            "name": "Model Interpretability",
            "description": "Understand what features a model has learned and what concepts they represent",
            "example": "Identify that centroid 42 corresponds to 'negation words' with 0.87 confidence",
        },
        {
            "name": "Debugging & Analysis",
            "description": "Diagnose why a model makes certain predictions by examining feature activations",
            "example": "Find that misclassified examples strongly activate centroid 15 (sentiment confusion)",
        },
        {
            "name": "Feature Discovery",
            "description": "Discover emergent features that weren't explicitly trained for",
            "example": "Uncover that centroid 73 represents 'programming syntax' despite general text training",
        },
        {
            "name": "Model Comparison",
            "description": "Compare different models by analyzing their learned features",
            "example": "Model A has 12 syntax-focused centroids vs Model B's 8, suggesting different representations",
        },
        {
            "name": "Training Insights",
            "description": "Monitor feature development during training",
            "example": "Track how centroid explanations stabilize and confidence increases over epochs",
        },
    ]

    for i, use_case in enumerate(use_cases, 1):
        print(f"{i}. {use_case['name']}")
        print(f"   {use_case['description']}")
        print(f"   Example: {use_case['example']}")
        print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("K-MEANS AUTOINTERPRETABILITY SYSTEM")
    print("Demonstration of Concepts and Workflow")
    print("=" * 80)

    demo_explanation_structure()
    demo_cache_operations()
    demo_validation_examples()
    demo_sentence_analysis_pipeline()
    demo_use_cases()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80 + "\n")

    print("System Overview:")
    print("  ✓ Centroid explanations with confidence scores")
    print("  ✓ Persistent cache for efficient repeated queries")
    print("  ✓ Top-k validation example matching")
    print("  ✓ LLM-based explanation generation")
    print("  ✓ Token-level analysis pipeline")
    print("  ✓ Meta-model interpretation")
    print("\nImplementation: exp/kmeans_autointerp.py")
    print("Tests: test/test_kmeans_autointerp.py\n")


if __name__ == "__main__":
    main()
