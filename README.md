# **Denoising With Mamba**
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache-2.0-brightgreen)](LICENSE)
[![TikTok TechJam 2025: 🥈 1st Runner Up](https://img.shields.io/badge/TikTok%20TechJam%202025-%F0%9F%A5%88%201st%20Runner%20Up-gold)](https://devpost.com/software/denoising-reviews-with-mamba)

## **TikTok TechJam 2025: 🥈 1st Runner Up**
> We are thrilled to announce that our project has managed to get 2nd place in the TikTok TechJam 2025!  
> 🔗 [View on Devpost](https://devpost.com/software/denoising-reviews-with-mamba)  

# **Overview**
The task of review classification requires models that are both highly accurate and scalable to large volumes of user-generated content. Our solution is a two-stage pipeline consisting of a Nomic embedder (encoder) [[1]] and a modified Mamba-based decoder [[2]]. This architecture was designed to balance computational efficiency, accuracy and scalability.

## **1. Model Architecture**

### **1.1 Embedding Stage**
We employ the open-source Nomic embedder to transform the reviews and location's metadata into dense semantic vectors. The review texts are outputted as a sequence of dense vectors while the location's metadata is returned as a singular vector.

- **Rationale:** Leaving the review texts as a sequence of dense vectors allows our decoder to map the relationships between tokens, allowing for greater insights to be learnt rather than from just a singular vector. Furthermore, we hypothesise that its inherent semantic optimisation would allow it to better capture relationships between locations and reviews unlike traditional BERT encoders.
- **Efficiency:** Location embeddings are cached and reused, and are only recalculated when the underlying metadata changes. This avoids the redundant computations and reduces training/inference cost.
- **Scalability:** Because embeddings can be pre-computed and stored, the system can scale more easily as less computational resources are required given that the computation of location embeddings does not occur unless its metadata changes. 

### **1.2 Decoding Stage** 
Our decoder is based on the Mamba architecture, a recent state space model (SSM) that replaces attention with a more efficient mechanism for modelling sequential dependencies.

**1.2.1 Mamba Advantages**
- **Computational Advantage:** While Transformer-based decoders scale at O(n^2) in sequence length, Mamba achieves O(n) complexity. This results in faster inference over long sequences.
- **Long-Range Dependencies:** As an SSM, Mamba excels at capturing long-distance contextual relationships in sequences. Transferring this to text, it allows us to better model the relevancy in a sequence of text.

**1.2.2 Modifications to Mamba**
However the Mamba architecture detailed in the paper does not allow for context injection which is what our current problem requires - the fusion of the location's metadata with the review. We overcame this through modifying Mamba's selection function from just a traditional Multi-Layer Perceptron (MLP) Layer to using a Feature-wise Linear Modulation (FiLM) [[3]] selection function.

- **Incorporating FiLM:** We use FiLM by passing our context vector into a MLP layer to generate gamma and beta parameters. We then transform the output of the original MLP over the input sequence using gamma and beta. This transformation allows us to incorporate our location's metadata into the input sequence selection function. Thus allowing us to ensure that the review can be classified while taking into account contextual location-specific factors.
 
## **2. Data Collection and Labelling**

### **2.1 Relevant Reviews**
We sourced reviews from Google Local Data (2021) [[4]], Kaggle as well as Hugging Face.

- **Label Integrity:** We employed Mistral Nemo [[5]] to filter and categories reviews as relevant or not relevant. At this step, our priority was to ensure a clean positive set and thus we used prompt engineering to enforce a strict relevance criteria, avoiding the contamination of the relevant class with noisy or borderline examples.

### **2.2 Irrelevant Reviews**
In order to easily generate irrelevant reviews, we simply shuffled the locations of the relevant reviews, ensuring that the shuffling will ensure no review would fall back under the same location category. We then used Mistral Nemo again in order to filter out only the reviews that are truly irrelevant from this new set. This method allowed us to increase the odds in which the LLM would find irrelevant reviews, saving us time and compute.


### **2.2 Non-Visitor Reviews and Advertisments**
Since non-visitor reviews and advertisments are often underrepresented, we generated them synthetically using the same 12B LLM. However, in order to ensure that our model does not overfit on the LLM's distributions, various techniques were used.

- **Context Injection:** We gave the LLM unique context for each prompt, either symbolising reviews to paraphase or locations to generate adverts for. These context would make up at least 80% of each prompt, thus allowing us to generate a diverse and varied set of data.
- **Randomised Parameters:** The LLM's temperature was randomised for each pass between 0.85 and 0.95, allowing for diverse outputs and the LLM's TopP was also set to 0.95 to ensure realistic reviews generated.

## **3. Training**
The model was trained over a small number of epochs, 10, due to computational resource restrictions. It used an AdamW optimiser, the CosineAnnealingLR scheduler and Binary Cross-Entropy Loss.

- **CosineAnnealingLR Scheduler:** As the model's loss was low over the training data, it used the scheduler to ensure that it did not overshoot the minima. 

## **4. Results**
Our model achieved **92%** accuracy and F1 score, outperforming our baseline of a cross-attention Transformer by +1% absolute improvement. Importantly, these gains were achieved while using fewer computational resources due to the efficiency of the Mamba decoder.

## **5. Additional Tests**
We conducted additional tests as well in order to validate our approach. It's results can be found below.

| Models                     | Accuracy |   F1   | ROC AUC |
|-----------------------------|----------|--------|---------|
| BERT – Mamba FiLM           | 0.83     | 0.827  | 0.949   |
| Nomic – Transformer         | 0.915    | 0.914  | 0.984   |
| Nomic – Mamba w/o FiLM      | 0.791    | 0.741  | 0.914   |
| **Nomic – Mamba FiLM (Ours)** | **0.929** | **0.929** | **0.989** |

> *Models are represented as Encoder - Decoder*  
> *The Transformer head used was 2.2M params while the Mamba FiLM head was 1.8M Params*  

- **BERT - Mamba FiLM:** This test was conducted to show how the choice to use a Nomic Embedder was valid and supports our hypothesis that its semantic training allows it to aid in classification better.
- **Nomic - Mamba w/o FiLM:** This shows how the FiLM selection function was critical to the function of the model to discern reviews from relevant to irrelevant. It proves that our model is not simply learning patterns in the reviews but rather using the modulation to learn how the reviews are linked to locations to classify from there.

## **6. Key Contributions**
- **Efficient Two-Stage Architecture:** Combining Nomic embeddings with a Mamba-based decoder yields much promise in terms of performance and at lower computational cost.
- **Modified Mamba with FiLM:** Our decoder is uniquely tailored to integrate location metadata directly into the selection mechanism, enhancing contextual awareness.
- **Data Generation Strategy:** A seperate pipeline for generating non-visitor reviews and advertisments via context injection for prompts and random model parameters, ensuring high dataset quality and robustness.

## **7. Conclusion**
This review relevance classification pipeline demonstrates that Mamba can outperform cross-attention based architectures at the same task of incorporating contextual information into the input sequence. By combining efficient embedding reuse, FiLM selection functions into state space decoding and innovating data generation strategies, we managed to achieve a robust, scalable, and highly accuracy classifier. 

[1]: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
[2]: https://arxiv.org/abs/2405.21060
[3]: https://arxiv.org/abs/1709.07871
[4]: https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/
[5]: https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
