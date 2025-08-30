### **Overview**
The task of review classification requires models that are both highly accurate and scalable to large volumes of user-generated content. Our solution is a two-stage pipeline consisting of Nomic embedder (encoder) [1] and a modified-Mamba based decoder [2]. This architecture was designed to balance computational efficiency, accuracy and scalability.

#### **1. Model Architecture**

##### **1.1 Embedding Stage**
We employ the open-source Nomic embedder to transform the reviews and location's metadata into dense semantic vectors. The review texts are outputted as a sequence of dense vectors while the location's metadata is returned as a singular vector.

⦁	**Rationale:** Leaving the review texts as a sequence of dense vectors allows our decoder to map the relationships between tokens, allowing for greater insights to be learnt rather than from just a singular vector.
⦁	**Efficiency:** Location embeddings are cached and reused, and are only recalculated when the underlying metadata changes. This avoids the redundant computations and reduces training/inference cost.
⦁	**Scalability:** Because embeddings can be pre-computed and stored, the system can scale more easily as less computational resources are required given that the computation of location embeddings does not occur unless its metadata changes. 

##### **1.2 Decoding Stage** 
Our decoder is based on the Mamba architecture, a recent state space model (SSM) that replaces attention with a more efficient mechanism for modelling sequential dependencies.

**1.2.1 Mamba Advantages**
⦁	**Computational Advantage:** While Transformer-based decoders scale at O(n^2) in sequence length, Mamba achieves O(nlogn) complexity. This results in faster inference and Mamba is also known for its lower memory footprint, making it highly suitable for production deployment.
⦁	**Long-Range Dependencies:** As an SSM, Mamba excels at capturing long-distance contextual relationships in sequences. Transferring this to text, it allows us to better model the relevancy in a sequence of text.

**1.2.2 Modifications to Mamba**
However the Mamba architecture detailed in the paper does not allow for context injection which is what our current problem requires - the fusion of the location's metadata with the review. We overcame this through modifying Mamba's selection function from a traditional Multi-Layer Perceptron (MLP) Layer to using a Feature-wise Linear Modulation (FiLM) [3] selection function.

⦁	**Incorporating FiLM:** We use FiLM by passing our context vector into a MLP layer to generate gamma and beta parameters. We then transform the output of the original MLP over the input sequence using gamma and beta. This transformation allows us to incorporate our location's metadata into the input sequence selection function. Thus allowing us to ensure that the review can be classified while taking into account contextual location-specific factors.
 
#### **2. Data Collection and Labelling**

##### **2.1 Relevant Reviews**
We sourced reviews from Google Local Data (2021) [4].

⦁	**Label Integrity:** We employed Mistral 7B Instruct [5] to filter and categories reviews as relevant or not relevant. At this step, our priority was to ensure a clean positive set and thus we used prompt engineering to enforce a strict relevance criteria, avoiding the contamination of the relevant class with noisy or borderline examples.

##### **2.2 Irrelevant Reviews**
Since irrelevant examples are often underrepresented, we generated them synthetically using the same 7B LLM.

⦁	**Prompt Variations:** Six different prompts were used, covering both positive and negative reviews for each policy listed.
⦁	**Deterministic Generation:** The LLM was set to deterministic mode to ensure stable output formats, aligned with our pipeline.
⦁	**Diversification Techniques:** We used category injection, the process of inserting targeted categories into the prompts, to enforce semantic variability as well as ensuring all categories were well represented. Furthermore, we also used randomised names at the start of every prompt to prevent repetitive output patterns. The random entity injections altered the sequences and attention distributions such that variable review formats were generated.

#### **3. Training**
The model was trained over a small number of epochs, 10, due to computational resource restrictions. It used an AdamW optimiser, the CosineAnnealingLR scheduler and Binary Cross-Entropy Loss.

⦁	**CosineAnnealingLR Scheduler:** As the model's loss was extremely low over the training data, it used the scheduler to ensure that it did not overshoot the minima. 

#### **4. Results**
Our model achieved **97% accuracy and F1 scoe, outperforming our baseline of a cross-attention Transformer by +1% absolute improvement. Importantly, these gains were achieved while using fewer computational resources due to the efficiency of the Mamba decoder.

#### **5. Additional Tests**
We conducted additional tests in order to see the importance of our FiLM selection function as well as the contextual information that the location's metadata provides. With the FiLM selection function, our model's validation accuracy managed to reach 97% while without, it drops to 55% meaning that the model could not learn any meaningful patterns. It can be taken that the model is unable to discern relevant reviews form irrelevant reviews without the location's metadata, showing the reliance of the model on the specific location context to make conclusions. These results support our choice in the modification of the selection function, showing how the model is also not simply learning patterns in the reviews but rather learning how the reviews are linked to the locations and then classifying from there.

#### **6. Key Contributions**
⦁	**Efficient Two-Stage Architecture:** Combining Nomic embeddings with a Mamba-based decoder yields much promise in terms of performance and at lower computational cost.
⦁	**Modified Mamba with FiLM:** Our decoder is uniquely tailored to integrate location metadata directly into the selection mechanism, enhancing contextual awareness.
⦁	**Data Generation Strategy:** A different approach for generating irrelevant reviews via deterministic LLM prompts with category + entity diversification, ensuring high dataset quality and robustness.

#### **7. Conclusion**
This review relevance classification pipeline demonstrates that Mamba can outperform cross-attention based architectures at the same task of incorporating contextual information into the input sequence. By combining efficient embedding reuse, FiLM selection functions into state space decoding and innovating data generation strategies, we managed to achieve a robust, scalable, and highly accuracy classifier. 

#### **References**
[1]: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
[2]: https://arxiv.org/abs/2405.21060
[3]: https://arxiv.org/abs/1709.07871
[4]: https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/
[5]: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3