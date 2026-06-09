# AI 工程术语表

## A

### Agent
- **人们怎么说：** "An autonomous AI that thinks and acts on its own"
- **实际是什么意思：** A while loop where an LLM decides what tool to call next, executes it, sees the result, and repeats
- **为什么这么叫：** Borrowed from philosophy — an "agent" is anything that can act in the world. In AI, it just means "LLM + tools + loop"

### Attention
- **人们怎么说：** "How the AI focuses on important parts"
- **实际是什么意思：** A mechanism where every token computes a weighted sum of all other tokens' values, with weights determined by how relevant they are (via dot product of query and key vectors)
- **为什么这么叫：** The 2017 paper "Attention Is All You Need" named it by analogy to human selective attention

### Alignment
- **人们怎么说：** "Making AI safe"
- **实际是什么意思：** The technical challenge of making an AI system's behavior match human intentions, values, and preferences, including edge cases the designer didn't anticipate

### Autoregressive
- **人们怎么说：** "The AI generates one word at a time"
- **实际是什么意思：** A model that predicts the next token conditioned on all previous tokens, then feeds that prediction back as input for the next step. GPT, LLaMA, and Claude are all autoregressive.

### Activation Function
- **人们怎么说：** "The nonlinear thing between layers"
- **实际是什么意思：** A function applied after each linear layer that introduces nonlinearity. Without it, stacking any number of linear layers collapses to a single linear transformation. ReLU, GELU, and SiLU are the most common. The choice directly affects whether gradients flow during training.

### Adam (Optimizer)
- **人们怎么说：** "The default optimizer"
- **实际是什么意思：** Adaptive Moment Estimation. Combines momentum (first moment) with adaptive learning rates per parameter (second moment). Has bias correction for early steps. Works well across most tasks without much tuning.

### AdamW
- **人们怎么说：** "Adam but better"
- **实际是什么意思：** Adam with decoupled weight decay. In standard Adam, L2 regularization gets scaled by the adaptive learning rate per parameter, which is not what you want. AdamW applies weight decay directly to the weights, independent of the gradient statistics. The default optimizer for training transformers.

### Autograd
- **人们怎么说：** "Automatic gradients"
- **实际是什么意思：** A system that records operations on tensors and automatically computes gradients via reverse-mode differentiation. PyTorch's autograd builds a computation graph on-the-fly (dynamic graph), while JAX uses function transformations (grad). This is what makes backpropagation practical -- you write the forward pass, and the framework computes all the derivatives.

## B

### Batch Size
- **人们怎么说：** "How many examples at once"
- **实际是什么意思：** The number of training examples processed in one forward/backward pass before updating weights. Larger batches give more stable gradient estimates but use more memory. Typical values: 32-512 for training, larger for inference. Batch size interacts with learning rate -- double the batch, double the LR (linear scaling rule).

### Backpropagation
- **人们怎么说：** "How neural networks learn"
- **实际是什么意思：** An algorithm that computes how much each weight contributed to the error by applying the chain rule backward through the network, then adjusts weights proportionally
- **为什么这么叫：** Errors propagate backward from output to input, layer by layer

## C

### Context Window
- **人们怎么说：** "How much the AI can remember"
- **实际是什么意思：** The maximum number of tokens (input + output) that fit in a single API call. Not memory — it's a fixed-size buffer that resets every call

### Chain of Thought (CoT)
- **人们怎么说：** "Making the AI think step by step"
- **实际是什么意思：** A prompting technique where you ask the model to show its reasoning steps, which improves accuracy on multi-step problems because each step conditions the next token generation

### CNN (Convolutional Neural Network)
- **人们怎么说：** "Image AI"
- **实际是什么意思：** A neural network that uses convolution operations (sliding filters over the input) to detect local patterns. Stacking convolutions detects increasingly complex features: edges, textures, objects.

### CUDA
- **人们怎么说：** "GPU programming"
- **实际是什么意思：** NVIDIA's parallel computing platform. Lets you run matrix operations on thousands of GPU cores simultaneously. PyTorch and TensorFlow use CUDA under the hood.

### Chunking
- **人们怎么说：** "Splitting documents into pieces"
- **实际是什么意思：** Breaking text into segments before embedding for retrieval. Chunk size determines the granularity of search results. Too small: loses context. Too large: dilutes relevance. Common strategies: fixed-size with overlap, sentence-based, or semantic splitting. Typical chunk size: 256-512 tokens with 10-20% overlap.

### Contrastive Learning
- **人们怎么说：** "Learning by comparison"
- **实际是什么意思：** Training by pulling similar pairs closer and pushing dissimilar pairs apart in embedding space. CLIP uses this: matching image-text pairs vs non-matching ones.

### Cosine Similarity
- **人们怎么说：** "How similar two vectors are"
- **实际是什么意思：** The cosine of the angle between two vectors: dot(a, b) / (||a|| * ||b||). Ranges from -1 (opposite) to 1 (identical direction). Ignores magnitude, only cares about direction. The standard similarity metric for embeddings and semantic search.

### Cross-Entropy
- **人们怎么说：** "The classification loss"
- **实际是什么意思：** Measures the difference between two probability distributions. For classification: -sum(y_true * log(y_pred)). For language models: the negative log probability of the correct next token. Lower is better. Perplexity is just exp(cross-entropy).

## D

### Data Augmentation
- **人们怎么说：** "Making more training data"
- **实际是什么意思：** Creating modified copies of existing data (rotate images, add noise, paraphrase text) to increase training set diversity without collecting new data. Reduces overfitting.

### Decoder
- **人们怎么说：** "The output part"
- **实际是什么意思：** In transformers, a decoder uses causal (masked) self-attention so each position can only attend to earlier positions. GPT is decoder-only. BERT is encoder-only. T5 is encoder-decoder.

### Diffusion Model
- **人们怎么说：** "AI that generates images from noise"
- **实际是什么意思：** A model trained to reverse a gradual noising process — it learns to predict and remove noise, and at generation time starts from pure noise and iteratively denoises

### DPO (Direct Preference Optimization)
- **人们怎么说：** "A simpler RLHF"
- **实际是什么意思：** A training method that skips the reward model entirely — it directly optimizes the language model to prefer the better response in pairs of human preferences

### Dropout
- **人们怎么说：** "Randomly turning off neurons"
- **实际是什么意思：** During training, randomly set a fraction of activations to zero. Forces the network to not rely on any single neuron. Turned off during inference. Simple but effective regularization.

## E

### Eigenvalue
- **人们怎么说：** "Some math thing for PCA"
- **实际是什么意思：** For a matrix A, an eigenvalue lambda satisfies Av = lambda*v for some vector v. It tells you how much the matrix scales vectors in that direction. Large eigenvalues = directions of high variance in your data.

### Embedding
- **人们怎么说：** "Some AI magic that turns words into numbers"
- **实际是什么意思：** A learned mapping from discrete items (words, images, users) to dense vectors in continuous space, where similar items end up close together
- **为什么这么叫：** The items are "embedded" in a geometric space where distance has meaning

### Encoder
- **人们怎么说：** "The input part"
- **实际是什么意思：** In transformers, an encoder uses bidirectional self-attention so each position can attend to all positions. BERT is encoder-only. Good for understanding tasks (classification, NER) but not generation.

### Epoch
- **人们怎么说：** "One pass through the data"
- **实际是什么意思：** Exactly that. One complete pass through every example in the training set. Multiple epochs = seeing the data multiple times. More epochs can improve learning but risks overfitting.

## F

### Feature
- **人们怎么说：** "A column in your data"
- **实际是什么意思：** An individual measurable property of the data. In classical ML, you engineer features by hand. In deep learning, the network learns features automatically from raw data.

### Few-Shot
- **人们怎么说：** "Give the AI some examples first"
- **实际是什么意思：** Including a small number of input-output examples in the prompt before asking the model to perform a task. Typically 3-5 examples. The model pattern-matches on these examples to understand the desired format and behavior. Contrast with zero-shot (no examples) and fine-tuning (thousands of examples baked into weights).

### Fine-tuning
- **人们怎么说：** "Training the AI on your data"
- **实际是什么意思：** Starting with a pre-trained model's weights and continuing training on a smaller, task-specific dataset. Only updates existing weights, doesn't add new knowledge from scratch

### Function Calling
- **人们怎么说：** "AI that can use tools"
- **实际是什么意思：** A structured way for LLMs to request execution of external functions. You define tools with JSON Schema descriptions, the model outputs a structured JSON object specifying which function to call with what arguments, your code executes it, and the result goes back to the model. Not the same as agents -- function calling is the mechanism, agents are the loop.

## G

### Guardrails
- **人们怎么说：** "Safety filters for AI"
- **实际是什么意思：** Input/output validation layers around an LLM that detect and block harmful content, prompt injection attempts, PII leakage, or off-topic responses. Typically a pipeline: input filter -> LLM -> output filter. Can be rule-based (regex, keyword lists) or model-based (classifier that scores safety).

### GPT
- **人们怎么说：** "ChatGPT" or "The AI"
- **实际是什么意思：** Generative Pre-trained Transformer — a specific architecture that predicts the next token using a decoder-only transformer trained on large text corpora
- **为什么这么叫：** Generative (produces text), Pre-trained (trained once on large data, then adapted), Transformer (the architecture)

### GAN (Generative Adversarial Network)
- **人们怎么说：** "Two AIs fighting each other"
- **实际是什么意思：** A generator network tries to create realistic data while a discriminator network tries to tell real from fake. They train together: the generator gets better at fooling the discriminator, and the discriminator gets better at detecting fakes.

### Gradient
- **人们怎么说：** "The slope"
- **实际是什么意思：** A vector of partial derivatives pointing in the direction of steepest increase. In ML, you go opposite to the gradient (gradient descent) to minimize the loss.

### Gradient Descent
- **人们怎么说：** "How AI improves"
- **实际是什么意思：** An optimization algorithm that adjusts parameters in the direction that reduces the loss function most steeply, like walking downhill in a high-dimensional landscape

## H

### Hyperparameter
- **人们怎么说：** "Settings you tune"
- **实际是什么意思：** Values set before training that control the training process itself: learning rate, batch size, number of layers, dropout rate. Unlike model parameters (weights), these aren't learned from data.

### Hallucination
- **人们怎么说：** "The AI is lying" or "making things up"
- **实际是什么意思：** The model generates plausible-sounding text that isn't grounded in its training data or the given context — it's pattern-completing, not fact-retrieving

## I

### Inference
- **人们怎么说：** "Running the AI"
- **实际是什么意思：** Using a trained model to make predictions on new data. No weight updates happen. This is what you do in production: send input, get output.

### Inductive Bias
- **人们怎么说：** Never heard of it
- **实际是什么意思：** The assumptions built into a model's architecture. CNNs assume local patterns matter (convolution). RNNs assume order matters (sequential processing). Transformers assume everything might relate to everything (attention). The right bias helps the model learn faster from less data.

### JAX
- **人们怎么说：** "Google's ML framework"
- **实际是什么意思：** A NumPy-compatible library that adds automatic differentiation (grad), JIT compilation (jit), automatic vectorization (vmap), and multi-device parallelism (pmap). Unlike PyTorch's object-oriented style, JAX is purely functional -- no hidden state, no in-place mutation. Used by Google DeepMind for AlphaFold, Gemini, and large-scale research.

## K

### KV Cache
- **人们怎么说：** "Makes inference faster"
- **实际是什么意思：** During autoregressive generation, caching the key and value matrices from previous tokens so you don't recompute them at each step. Trades memory for speed. Essential for fast LLM inference.

## L

### Latent Space
- **人们怎么说：** "The hidden representation"
- **实际是什么意思：** A compressed, learned representation space where similar inputs map to nearby points. Autoencoders, VAEs, and diffusion models all work in latent space. It's lower-dimensional than the input but captures the important structure.

### Learning Rate
- **人们怎么说：** "How fast the AI learns"
- **实际是什么意思：** A scalar that controls step size during gradient descent. Too high: overshoots the minimum and diverges. Too low: converges too slowly or gets stuck. The single most important hyperparameter.

### LLM (Large Language Model)
- **人们怎么说：** "AI" or "the brain"
- **实际是什么意思：** A transformer-based neural network trained to predict the next token in a sequence, with billions of parameters, trained on internet-scale text data

### LoRA (Low-Rank Adaptation)
- **人们怎么说：** "Efficient fine-tuning"
- **实际是什么意思：** Instead of updating all weights, insert small low-rank matrices alongside the original weights. Only these small matrices are trained, reducing memory by 10-100x

### Loss Function
- **人们怎么说：** "How wrong the AI is"
- **实际是什么意思：** A function that measures the gap between predicted and actual output. Training minimizes this function. MSE for regression, cross-entropy for classification, contrastive loss for embeddings. The choice of loss function defines what "good" means to the model.

## M

### Mixed Precision
- **人们怎么说：** "Training trick for speed"
- **实际是什么意思：** Using float16 for forward pass and most operations (faster, less memory) but keeping float32 for gradient accumulation and weight updates (more precise). Gets 2x speedup with negligible accuracy loss.

### MoE (Mixture of Experts)
- **人们怎么说：** "Only part of the model runs"
- **实际是什么意思：** A model with many "expert" subnetworks where a routing mechanism sends each input to only a few experts. The full model is huge but each forward pass is cheap because most experts are skipped. Mixtral and GPT-4 use this.

### MCP (Model Context Protocol)
- **人们怎么说：** "A way for AI to use tools"
- **实际是什么意思：** An open protocol (JSON-RPC over stdio/HTTP) that standardizes how AI applications connect to external data sources and tools, with typed schemas for tools, resources, and prompts

## N

### NaN (Not a Number)
- **人们怎么说：** "Training crashed"
- **实际是什么意思：** A floating-point value indicating undefined results (0/0, inf-inf). In training, NaN loss usually means: learning rate too high, exploding gradients, log of zero, or division by zero. Always the first thing to check when training fails.

### Normalization
- **人们怎么说：** "Scaling the data"
- **实际是什么意思：** Adjusting values to a standard range. Batch normalization normalizes across a batch. Layer normalization normalizes across features. Both stabilize training and allow higher learning rates.

## O

### Overfitting
- **人们怎么说：** "The model memorized the data"
- **实际是什么意思：** The model performs well on training data but poorly on unseen data. It learned the noise, not the signal. Fix with: more data, regularization (dropout, weight decay), early stopping, data augmentation, simpler model.

### Optimizer
- **人们怎么说：** "The thing that updates weights"
- **实际是什么意思：** An algorithm that uses gradients to update model parameters. SGD is the simplest. Adam is the most common. Each optimizer has different properties: convergence speed, memory usage, sensitivity to hyperparameters.

## P

### Parameter
- **人们怎么说：** "Model size"
- **实际是什么意思：** A learnable value in the model, typically a weight or bias. "7B parameters" means 7 billion learnable numbers. Each float32 parameter takes 4 bytes, so 7B parameters = 28GB of memory just for the weights.

### Perplexity
- **人们怎么说：** "How confused the model is"
- **实际是什么意思：** The exponential of the average cross-entropy loss. Lower is better. A perplexity of 10 means the model is as uncertain as if it were choosing uniformly among 10 tokens at each step.

### Precision & Recall
- **人们怎么说：** "Accuracy metrics"
- **实际是什么意思：** Precision = of items you flagged, how many were correct. Recall = of all correct items, how many did you find. They trade off: catching every spam email (high recall) means more false alarms (low precision). F1 score is their harmonic mean. Use precision when false positives are costly, recall when false negatives are costly.

### Prompt Engineering
- **人们怎么说：** "Talking to AI the right way"
- **实际是什么意思：** Designing the input text to reliably produce desired outputs -- including system prompts, few-shot examples, format instructions, and chain-of-thought triggers

### Prompt Injection
- **人们怎么说：** "Hacking the AI with words"
- **实际是什么意思：** An attack where malicious text in the input overrides the system prompt or instructions. Direct injection: user types "Ignore previous instructions." Indirect injection: a retrieved document contains hidden instructions. The LLM equivalent of SQL injection. No complete solution exists -- defense is layers of input validation, output filtering, and privilege separation.

## Q

### QLoRA
- **人们怎么说：** "LoRA but cheaper"
- **实际是什么意思：** Quantized LoRA. Keeps the frozen base model weights in 4-bit precision (NF4 format) while training LoRA adapters in 16-bit. Reduces memory by another 3-4x compared to standard LoRA. A 7B model that needs 14GB with LoRA fits in 4-6GB with QLoRA. Quality is within 1% of full fine-tuning on most benchmarks.

## R

### RAG (Retrieval-Augmented Generation)
- **人们怎么说：** "AI that can search"
- **实际是什么意思：** A pattern where you retrieve relevant documents from a knowledge base (using embedding similarity), stuff them into the prompt, and let the LLM answer based on that context
- **为什么这么叫：** Retrieval (find documents) + Augmented (add to prompt) + Generation (LLM writes the answer)

### RLHF (Reinforcement Learning from Human Feedback)
- **人们怎么说：** "How they make AI helpful"
- **实际是什么意思：** A training pipeline: (1) collect human preferences on model outputs, (2) train a reward model on those preferences, (3) use PPO to optimize the LLM to produce higher-reward outputs

### Quantization
- **人们怎么说：** "Making the model smaller"
- **实际是什么意思：** Reducing the precision of model weights from float32 (4 bytes) to int8 (1 byte) or int4 (0.5 bytes). Trades a small amount of accuracy for 4-8x less memory and faster inference. GPTQ, AWQ, and GGUF are common formats.

### ReLU
- **人们怎么说：** "Activation function"
- **实际是什么意思：** Rectified Linear Unit: f(x) = max(0, x). The simplest non-linear activation. Fast to compute, doesn't saturate for positive values. Used everywhere because it works and is cheap. Variants: LeakyReLU, GELU, SiLU.

### ROUGE
- **人们怎么说：** "Summarization metric"
- **实际是什么意思：** Recall-Oriented Understudy for Gisting Evaluation. Measures overlap between generated text and reference text. ROUGE-1 counts unigram matches, ROUGE-2 counts bigram matches, ROUGE-L finds the longest common subsequence. Cheap to compute but only measures surface similarity -- two sentences with the same meaning but different words score poorly.

## S

### Semantic Search
- **人们怎么说：** "Smart search that understands meaning"
- **实际是什么意思：** Finding documents by meaning rather than keyword matching. Embed the query and all documents into the same vector space, then return documents whose embeddings are closest to the query embedding. "payment failed" finds "transaction declined" even though they share no words. Powered by embedding models + vector databases.

### Streaming
- **人们怎么说：** "Seeing the response appear word by word"
- **实际是什么意思：** The LLM sends tokens as they are generated rather than waiting for the complete response. Uses Server-Sent Events (SSE) or WebSocket protocols. Reduces perceived latency from seconds to milliseconds for the first token. Essential for production chat interfaces. Each chunk contains a delta (partial token or word).

### Self-Attention
- **人们怎么说：** "How the model decides what to focus on"
- **实际是什么意思：** Each token computes query, key, and value vectors. Attention weight between two tokens = dot product of their query and key, scaled and softmaxed. Output = weighted sum of value vectors. Lets every token see every other token.

### SFT (Supervised Fine-Tuning)
- **人们怎么说：** "Teaching the model to follow instructions"
- **实际是什么意思：** Fine-tuning a pre-trained model on (instruction, response) pairs. The model learns to generate the response given the instruction. This is what turns a base model into a chat model.

### Softmax
- **人们怎么说：** "Turns numbers into probabilities"
- **实际是什么意思：** softmax(x_i) = exp(x_i) / sum(exp(x_j)). Transforms a vector of arbitrary real numbers into a probability distribution (all positive, sums to 1). Used in classification heads, attention weights, and anywhere you need probabilities.

### Swarm
- **人们怎么说：** "A bunch of AI agents working together like bees"
- **实际是什么意思：** Multiple agents sharing state and coordinating through message passing, with emergent behavior arising from simple individual rules rather than central control

## T

### System Prompt
- **人们怎么说：** "The AI's instructions"
- **实际是什么意思：** A special message at the start of a conversation that sets the model's behavior, persona, and constraints. Processed before user messages. Not visible to the user in most UIs. Defines what the model should and shouldn't do, its tone, format preferences, and domain focus. Different from user prompts -- system prompts are set by the developer.

### Tensor
- **人们怎么说：** "A multi-dimensional array"
- **实际是什么意思：** The fundamental data structure in deep learning frameworks. A 0D tensor is a scalar, 1D is a vector, 2D is a matrix, 3D+ is a tensor. In PyTorch and JAX, tensors track their computation history for automatic differentiation and can live on CPU or GPU. All neural network inputs, outputs, weights, and gradients are tensors.

### Token
- **人们怎么说：** "A word"
- **实际是什么意思：** A subword unit (typically 3-4 characters in English) produced by a tokenizer like BPE. "unbelievable" might be 3 tokens: "un" + "believ" + "able"

### Temperature
- **人们怎么说：** "Creativity setting"
- **实际是什么意思：** A scalar that divides logits before softmax. Temperature=1 is default. Higher = flatter distribution = more random outputs. Lower = sharper distribution = more deterministic. Temperature=0 is argmax (always pick the most likely token).

### Transfer Learning
- **人们怎么说：** "Using a pre-trained model"
- **实际是什么意思：** Taking a model trained on one task and adapting it to a different task. The early layers learn general features (edges, syntax patterns) that transfer. Only the later layers need task-specific training. This is why you can fine-tune BERT for any NLP task.

### Transformer
- **人们怎么说：** "The architecture behind modern AI"
- **实际是什么意思：** A neural network architecture that processes sequences using self-attention (letting every position attend to every other position) instead of recurrence, enabling massive parallelization
- **为什么这么叫：** It transforms input representations into output representations through attention layers

## U

### Underfitting
- **人们怎么说：** "The model isn't learning"
- **实际是什么意思：** The model is too simple to capture the patterns in the data. Training loss stays high. Fix with: more parameters, more layers, longer training, lower regularization, better features.

## V

### VAE (Variational Autoencoder)
- **人们怎么说：** "A generative model"
- **实际是什么意思：** An autoencoder that learns a smooth latent space by forcing the encoder output to follow a Gaussian distribution. You can sample from this distribution and decode to generate new data. The reparameterization trick makes it trainable via backpropagation.

### Vector Database
- **人们怎么说：** "A special database for AI"
- **实际是什么意思：** A database optimized for storing vectors (dense arrays of floats) and performing fast approximate nearest-neighbor search. The core operation in similarity search, RAG, and recommendation systems.

## W

### Weight
- **人们怎么说：** "What the model learned"
- **实际是什么意思：** A single number in a model's parameter matrix. A linear layer with input size 768 and output size 3072 has 768*3072 = 2,359,296 weights. Training adjusts each weight to minimize the loss function.

### Weight Decay
- **人们怎么说：** "Regularization"
- **实际是什么意思：** Adding a penalty proportional to the magnitude of weights to the loss function. Equivalent to L2 regularization. Prevents weights from growing too large. Typical value: 0.01-0.1.

## Z

### Zero-Shot
- **人们怎么说：** "No training needed"
- **实际是什么意思：** Using a model on a task it wasn't explicitly trained for, with no task-specific examples in the prompt. The model generalizes from pre-training. Works because large models have seen enough variety to handle new task formats.
