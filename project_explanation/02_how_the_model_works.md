# 2. How the model works

## 2.1 What a vision-language model is

A **vision-language model** is an AI model that takes an image plus a written instruction, and writes text as its answer.

Think of a careful assistant who looks at a receipt and types out what they see, following your instructions. The model does the same, but it learned how to do it from millions of images and texts during its original training.

## 2.2 The model we use: Qwen3-VL-4B-Instruct

| | |
|---|---|
| Made by | the Qwen team at Alibaba |
| Licence | Apache 2.0, which means it is free to use |
| Downloaded from | the official Qwen page on Hugging Face, a website that hosts AI models |
| "4B" | about 4 billion **parameters**. These are the numbers inside the model that were learned during its original training, also called **weights**. |
| "Instruct" | the version trained to follow instructions |
| "VL" | Vision-Language |

The model has two main parts:

```
 receipt image ──► VISION ENCODER ──► image tokens ─┐
                                                    ├──► LANGUAGE MODEL ──► answer text, one token at a time
 instruction text ──► text tokens ──────────────────┘
```

- The **vision encoder** turns the picture into a list of numbers the language model can understand.
- The **language model** reads those numbers together with the instruction, and writes the answer.

## 2.3 Tokens: how text and images are cut into pieces

The model does not read whole words or whole pictures. It reads **tokens**:

- **Text** is cut into small pieces. Common words are one token; rare words are split into several.
- **An image** is cut into tiles of 32 × 32 pixels, and **each tile becomes one image token**.

For example, receipt2:

| | |
|---|---|
| Original image | 439 × 1004 pixels |
| Size the model sees | 416 × 992 pixels |
| Image tokens | (416 ÷ 32) × (992 ÷ 32) = 13 × 31 = **403** |
| Instruction and chat markers | 43 tokens |
| Total the model reads | **446 tokens** |
| Answer written by the final model | 190 tokens |

## 2.4 Why big images are made smaller

More tokens mean more memory and more time. So images are shrunk until they have at most about 1,024 tiles, which is about 1 megapixel.

- **The biggest receipt**, receipt956, is 4961 × 7016 pixels (35 megapixels). It is shrunk to 832 × 1216 pixels, which is 988 image tokens.
- **Small images are never made bigger.** Their sides are only trimmed down to a multiple of 32. For example, receipt6 goes from 463 × 605 to 448 × 576.
- **314 of the 970 images** are already under 1 megapixel.

1 megapixel keeps the small print readable, at about 20 seconds per receipt on the Mac.

## 2.5 How the answer is written: one token at a time

The language model writes the answer **one token at a time**:

1. It looks at everything so far (image tokens, the instruction, and the answer written so far).
2. It gives every possible next token a probability, for example `"` 97%, `{` 2%, and so on.
3. It picks one token and adds it to the answer.
4. It repeats until it writes a special "end" token, or reaches the limit of 2,048 tokens.

We use **greedy decoding**: at step 3, always pick the most likely token. This makes the model **repeatable**: the same image always gives exactly the same answer. We checked this several times, and the outputs were identical byte for byte.

## 2.6 The instruction (prompt) and the chat format

The written instruction is called the **prompt**. Three prompts were used in this project; all are in `src/`:

| File | Used for | Length |
|---|---|---|
| `prompt_v1.txt` | the first run, before fine-tuning | 245 words, with simple field definitions |
| `prompt_v2.txt` | the improved run, before fine-tuning | 621 words, with detailed rules (chapter 5) |
| `prompt_ft.txt` | the fine-tuned model | 1 sentence |

The model expects a conversation, like a chat:
- **user:** [the image] + "Extract the information from this receipt as one JSON object with the keys ..."
- **assistant:** (the model writes its answer here)

The code builds this chat text with a function called `apply_chat_template`. In it, the image comes first and the instruction second.

## 2.7 MLX: running the model on the Mac

**MLX** is Apple's free library for running AI models on Apple Silicon chips (M1 to M5).

- **The GPU does the work.** The chip's graphics processor runs the model much faster than the main processor could.
- **Unified memory.** On these Macs the GPU and the main processor share the same 16 GB of memory, so a 4.8 GB model fits easily.
- **mlx-vlm.** We use a package called **mlx-vlm**, which knows how to load and run vision-language models like Qwen3-VL with MLX.

MLX also has a CPU-only version for Linux. That is how the project can run on an Ubuntu laptop, though more slowly (chapter 8).

## 2.8 8-bit: making the model smaller

The downloaded model stores each of its 4 billion numbers using 16 bits, which adds up to 8.3 GB. We converted it to **8 bits** per number:

- **What changes:** each number is stored with less precision. Groups of 64 numbers share one scale factor, which keeps the rounding error small.
- **Size:** 8.3 GB becomes **4.8 GB**, and it runs faster.
- **Accuracy:** it almost does not change.

It is like writing prices to the nearest paisa instead of with ten decimal places. The value is practically the same, but it takes less space.

This conversion is called **quantization**. Every run in this project used the same 8-bit model, `models/Qwen3-VL-4B-Instruct-8bit`.

**Checksums.** A **checksum** (sha256) is a long code calculated from a file's contents. If even one byte is different, the code is completely different. The README lists the checksums of the official weights and of the 8-bit conversion. On another computer you can check that your files are exactly the same.
