# K-Means Autointerpretability System

A comprehensive autointerpretability framework for understanding neural network activations through k-means clustering. This system provides natural language explanations for learned features with confidence scores, enabling deeper insight into model behavior.

## Overview

The K-Means Autointerpretability system consists of three main components:

1. **Explanation Cache**: Stores natural language descriptions of k-means centroids with confidence levels
2. **Validation Matching**: Identifies top-k examples that activate each centroid most strongly
3. **Analysis Pipeline**: Tokenizes sentences, runs them through the model, and generates interpretations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    K-Means Autointerpretability                 │
└─────────────────────────────────────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
   ┌────────▼────────┐ ┌──────▼───────┐ ┌────────▼────────┐
   │ Explanation     │ │  Validation  │ │    Analysis     │
   │     Cache       │ │   Matching   │ │    Pipeline     │
   └─────────────────┘ └──────────────┘ └─────────────────┘
            │                  │                  │
            │                  │                  │
   ┌────────▼──────────────────▼──────────────────▼────────┐
   │              Centroid Explanations                     │
   │   { centroid_id, explanation, confidence, examples }   │
   └────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Centroid Explanations with Confidence

Each k-means centroid is assigned a natural language explanation along with a confidence score:

```python
CentroidExplanation(
    centroid_id=42,
    explanation="This feature activates on punctuation marks and sentence boundaries",
    confidence=0.85,  # 0.0 to 1.0
    top_k_examples=["The cat sat on the mat.", "Hello, world!", ...],
    timestamp="2024-01-01T12:00:00",
    metadata={"layer_idx": 6, "activation_type": "layer_output"}
)
```

### 2. Lazy Computation with Caching

Explanations are generated on-demand and cached for future use:

- **First access**: Query top-k validation examples → Generate explanation via LLM → Cache result
- **Subsequent accesses**: Retrieve from cache instantly
- **Persistence**: Cache stored to disk for cross-session reuse

### 3. Validation Example Matching

For each centroid, the system identifies the validation examples that activate it most strongly:

```python
validation_examples = autointerp.get_top_k_validation_examples(
    centroid_id=5,
    k=10
)
# Returns: List[ValidationExample] with highest activation values
```

### 4. Token-Level Analysis

Analyze any sentence to see which features activate for each token:

```python
results = await autointerp.analyze_sentence(
    "The quick brown fox jumps over the lazy dog."
)
# Returns per-token analysis with explanations and confidence scores
```

### 5. Meta-Interpretation

Generate high-level interpretations of how the model processes text:

```python
interpretation = await autointerp.interpret_with_meta_model(
    "The quick brown fox jumps over the lazy dog."
)
# Returns: Natural language interpretation of model behavior
```

## Usage

### Basic Setup

```python
from exp.kmeans_autointerp import KMeansAutoInterp, load_kmeans_centroids
from nnterp import StandardizedTransformer

# Load model
model = StandardizedTransformer("gpt2", device_map="auto")

# Load k-means centroids
centroids, metadata = load_kmeans_centroids("path/to/centroids.pt")

# Initialize autointerpretability system
autointerp = KMeansAutoInterp(
    cache_dir="artifacts/kmeans_cache",
    model=model,
    centroids=centroids,
    validation_activations=validation_acts,  # Optional: pre-computed
    validation_tokens=validation_tokens,      # Optional: pre-computed
    layer_idx=6,
    activation_type="layer_output"
)
```

### Analyzing a Sentence

```python
import asyncio

# Analyze a sentence
results = await autointerp.analyze_sentence(
    "The quick brown fox jumps over the lazy dog.",
    k=10  # Number of validation examples to use
)

# Display results
for result in results:
    print(f"Token {result['position']}: {result['token_str']}")
    print(f"  Centroid: {result['centroid_id']}")
    print(f"  Explanation: {result['explanation']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print()
```

### Getting Centroid Explanations

```python
# Get or generate explanation for a centroid
explanation = await autointerp.get_or_generate_explanation(
    centroid_id=42,
    k=10,
    llm_client=None,  # Optional: LLM client for generation
    force_regenerate=False
)

print(f"Explanation: {explanation.explanation}")
print(f"Confidence: {explanation.confidence}")
print(f"Top examples: {explanation.top_k_examples}")
```

### Meta-Interpretation

```python
# Get high-level interpretation
interpretation = await autointerp.interpret_with_meta_model(
    "The quick brown fox jumps over the lazy dog.",
    k=10
)

print(interpretation)
```

## Data Structures

### CentroidExplanation

```python
@dataclass
class CentroidExplanation:
    centroid_id: int
    explanation: str
    confidence: float  # 0.0 to 1.0
    top_k_examples: list[str]
    timestamp: str
    metadata: dict[str, Any]
```

### ValidationExample

```python
@dataclass
class ValidationExample:
    text: str
    tokens: list[int]
    token_strings: list[str]
    activation: float
    token_position: int
```

## Cache Structure

Explanations are stored in JSON format:

```
cache_dir/
├── centroid_0.json
├── centroid_1.json
├── centroid_2.json
└── ...
```

Each file contains:

```json
{
  "centroid_id": 0,
  "explanation": "Punctuation and sentence boundaries",
  "confidence": 0.95,
  "top_k_examples": ["...", "...", "..."],
  "timestamp": "2024-01-01T12:00:00",
  "metadata": {
    "layer_idx": 6,
    "activation_type": "layer_output",
    "generation_method": "llm"
  }
}
```

## Pipeline Details

### Full Analysis Pipeline

The system implements a complete 3-step autointerpretability pipeline:

**Step 1**: Tokenize input and extract activations
```python
tokens = tokenizer(sentence)
activations = model.get_activations(tokens, layer_idx=6)
```

**Step 2**: Assign tokens to nearest centroids
```python
centroid_ids = assign_tokens_to_centroids(activations)
```

**Step 3**: Get or generate explanations
```python
for centroid_id in unique(centroid_ids):
    explanation = get_or_generate_explanation(centroid_id)
```

**Step 4**: Generate meta-interpretation
```python
interpretation = interpret_with_meta_model(token_analyses)
```

## LLM Integration

The system supports custom LLM clients for explanation generation. By default, it uses a heuristic approach, but you can provide your own LLM client:

```python
# Custom LLM client (pseudocode)
class CustomLLMClient:
    def generate(self, prompt: str) -> dict:
        # Call your LLM API
        response = api.complete(prompt)
        return {
            "explanation": response.explanation,
            "confidence": response.confidence
        }

llm_client = CustomLLMClient()
explanation = await autointerp.generate_explanation(
    centroid_id=42,
    examples=top_k_examples,
    llm_client=llm_client
)
```

## Example Output

### Token-Level Analysis

```
Pos  Token        C-ID  Conf   Act     Explanation
------------------------------------------------------------------------------
0    The          2     0.88   1.234   Determiners and articles
1     quick       7     0.75   0.987   Adjectives describing speed or intensity
2     brown       8     0.82   1.045   Color descriptors
3     fox         3     0.90   1.567   Animal names
4     jumps       5     0.85   1.423   Action verbs indicating motion
5     over        1     0.78   0.876   Prepositions of location/direction
6     the         2     0.88   1.198   Determiners and articles
7     lazy        7     0.75   0.934   Adjectives describing speed or intensity
8     dog         3     0.90   1.621   Animal names
9    .            0     0.95   1.789   Punctuation and sentence boundaries
```

### Meta-Interpretation

```
The model processes this sentence by activating distinct feature patterns for different
linguistic categories:

1. SYNTACTIC STRUCTURE: The sentence is clearly bounded by punctuation features 
   (centroid 0, 0.95 confidence). Determiners "the" consistently activate centroid 2
   (0.88 confidence), showing stable article recognition.

2. SEMANTIC CATEGORIES: The model distinguishes:
   - Animal nouns ("fox", "dog") → centroid 3 (0.90 confidence, high activation)
   - Motion verbs ("jumps") → centroid 5 (0.85 confidence)
   - Color adjectives ("brown") → centroid 8 (0.82 confidence)

This reveals hierarchical processing: the model first identifies syntactic roles,
then semantic categories. The consistent activation patterns demonstrate organized
internal representations.
```

## Use Cases

1. **Model Interpretability**: Understand what features a model has learned
2. **Debugging & Analysis**: Diagnose why models make certain predictions
3. **Feature Discovery**: Discover emergent features not explicitly trained
4. **Model Comparison**: Compare learned features across different models
5. **Training Insights**: Monitor feature development during training

## Testing

Run the test suite:

```bash
pytest test/test_kmeans_autointerp.py -v
```

Run the demo:

```bash
python scripts/demo_kmeans_autointerp_simple.py
```

## Files

- `exp/kmeans_autointerp.py` - Main implementation
- `test/test_kmeans_autointerp.py` - Comprehensive test suite
- `scripts/demo_kmeans_autointerp_simple.py` - Demonstration script
- `exp/KMEANS_AUTOINTERP_README.md` - This documentation

## Dependencies

- `torch` - For tensor operations and k-means
- `transformers` - For model and tokenizer
- `nnterp` - For standardized transformer interface
- `loguru` - For logging
- `tqdm` - For progress bars

## Performance Considerations

- **Caching**: Explanations are cached to avoid redundant LLM calls
- **Lazy Generation**: Explanations only generated when needed
- **Batch Processing**: Validation matching uses efficient batch operations
- **Memory**: Uses bfloat16 for activations to reduce memory usage

## Future Enhancements

- [ ] Support for multiple LLM providers (OpenAI, Anthropic, etc.)
- [ ] Parallel explanation generation
- [ ] Confidence calibration based on validation metrics
- [ ] Interactive visualization of token activations
- [ ] Explanation refinement based on human feedback
- [ ] Multi-layer analysis and feature tracking
- [ ] Automated centroid labeling from explanation patterns

## Citation

If you use this system in your research, please cite:

```bibtex
@software{kmeans_autointerp,
  title = {K-Means Autointerpretability System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo/moe-router-study}
}
```

## License

[Your License Here]
