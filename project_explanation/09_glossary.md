# 9. Glossary: every technical word, simply

Listed alphabetically.

| Word | Simple meaning |
|---|---|
| **Adapter** | The small file of LoRA corrections made by fine-tuning (`models/lora_v1/epoch2/adapters.safetensors`, 132 MB). It is used together with the base model. |
| **AdamW** | The method that decides how to change the trainable numbers at each training step, using the gradients. It is the standard choice. |
| **Amount** | In a line item, the price printed on that row. In scoring, all money values are compared as numbers. |
| **Apple Silicon** | Apple's own chips (M1, M2, M3, M4, M5). MLX's fast mode needs one of these. |
| **Baseline** | The first, simplest version, which every improvement is compared against. Here: the model as it is, with prompt v1. |
| **Bootstrap** | Estimating how much a score could vary by redrawing the test receipts at random many times (chapter 7.4). |
| **Chat template** | The fixed text layout the model expects: who is speaking (user or assistant), where the image goes, and where the answer starts. |
| **Checksum (sha256)** | A long code calculated from a file's contents. Identical files have identical codes; one changed byte gives a completely different code. |
| **Cleanup** | The formatting-only fixes applied to the model's answer, such as "RM6.69" → "6.69" (`clean_values`, chapter 3.4). |
| **Cosine schedule** | Lowering the learning rate along a smooth curve during training: fast at first, then gently. |
| **CPU** | The computer's main processor. It runs anything, but it is much slower than a GPU for AI models. |
| **CSV** | A simple table file (comma-separated values). It opens in Excel or Google Sheets. |
| **Decoding (greedy)** | How the model picks each next token. Greedy means always the most likely one, so the same input always gives the same answer. |
| **dev100** | 100 train receipts (every 6th one), never used for teaching. They are used to compare versions and pick the best (`src/dev100.txt`). |
| **Epoch** | One full pass over all training examples. We trained 2 epochs. |
| **Exact accuracy** | Text must match exactly, apart from capital letters and extra spaces. Punctuation counts. |
| **F1** | One number combining precision and recall for line items: 2 × P × R ÷ (P + R). |
| **Fine-tuning** | Training an existing model further on your own examples, so it becomes better at your task. |
| **Fuzzy match (95% similar)** | Text counts as correct if it is at least 95% similar to the label. |
| **Git, GitHub** | Git keeps the history of every change to the project's files. GitHub stores that history online (`github.com/mohit6603/ccp`). |
| **GPU** | The graphics processor. It does many calculations at once, so it runs AI models fast. |
| **Gradient** | For each trainable number, the direction (bigger or smaller) that would reduce the error. |
| **Gradient accumulation** | Adding up the gradients of several examples (here 4) before making one update. It saves memory. |
| **Gradient checkpointing** | Recomputing some intermediate results instead of storing them. It uses less memory and takes more time. |
| **Hallucination** | The model gives a value where the label is empty, which means it invented something. |
| **Hugging Face** | A website that hosts AI models. The official Qwen weights were downloaded from there. |
| **Image token** | One 32 × 32-pixel tile of the image, turned into numbers for the model. |
| **Inference** | Using a trained model to produce answers. Running `run_model.py` is inference. |
| **JSON** | A text format of `"name": value` pairs. Our output format. |
| **Label** | The correct answer for a receipt, in `receipt_labeled_text/`. |
| **Learning rate** | How big each training step is. Too big and training becomes unstable; too small and it learns slowly. |
| **Line item** | One product row on the receipt: description, quantity, unit price, amount. |
| **LoRA** | Low-Rank Adaptation: freeze the model and train only small add-on corrections (chapter 6.2). |
| **Loss** | A number for how wrong (surprised) the model is about the correct answer. Lower is better; e^(−loss) is the average probability of the correct token. |
| **Miss** | The model leaves a field empty although the label has a value. |
| **MLX / mlx-vlm** | MLX is Apple's library for running AI models. mlx-vlm is the add-on for vision-language models like Qwen3-VL. |
| **Natural order** | Sorting names by their numbers, so receipt2 comes before receipt10. |
| **Null** | JSON's word for "empty". Here it means the information is not on the receipt. |
| **Optimizer step** | One update of the trainable numbers (here after every 4 receipts). |
| **Overall accuracy** | The headline score: the 7 single-value fields, where company, address and tax number ignore spaces and punctuation, but every letter and digit must be right. |
| **Overfitting** | The model memorises its training examples instead of learning the pattern. You can see it when the validation loss goes up. |
| **Parameters (weights)** | The numbers inside the model that were learned in training. Qwen3-VL-4B has about 4 billion. |
| **Precision** | Of the items the model wrote, the share that are correct. |
| **Prompt** | The written instruction given to the model with the image (`src/prompt_*.txt`). |
| **Quantization (8-bit)** | Storing the model's numbers with fewer bits (8 instead of 16) to make it smaller and faster, with very little accuracy loss. |
| **Rank** | The width of the LoRA corrections (here 16). Bigger means more trainable numbers. |
| **Raw output** | The exact text the model wrote, before reading it as JSON (`outputs/<run>/raw/`). |
| **Recall** | Of the items printed on the receipt (in the label), the share the model got correct. |
| **Resume** | Continuing an interrupted run by repeating the same command. Finished receipts are skipped. |
| **Run** | One use of `run_model.py` with fixed settings and a name, for example test_final. |
| **safetensors** | The file format used for model weights and adapters. |
| **Schema** | The required shape of the answer: exactly these 8 keys, and these 4 keys per item. |
| **Scoring (evaluation)** | Comparing answers with labels to measure accuracy (`evaluate.py`). |
| **Split** | One of the groups of receipts: train or test. |
| **SROIE** | The public research collection of scanned Malaysian receipts that this dataset comes from. |
| **Swap** | When memory is full, the operating system moves some of it to the disk. It is slower, but lets bigger jobs finish. |
| **Test set** | The 370 receipts used once at the very end, to measure the final model fairly. |
| **Token** | A small piece of text, or one image tile, as the model reads and writes them. |
| **Train set** | The 600 receipts used for finding mistakes, improving, and teaching. |
| **Unit price** | The price of one unit of an item, filled only if printed separately. |
| **Validation loss** | The loss on the dev receipts, measured without learning. It shows whether training also helps on unseen receipts. |
| **venv (.venv)** | The project's own Python environment: a folder with exactly the right package versions installed. |
| **Vision encoder** | The part of the model that turns the image into image tokens. |
| **Vision-language model** | A model that reads an image plus text and writes text. |
| **Warm-up** | Starting training with a small learning rate and raising it over the first steps (here 10). |
