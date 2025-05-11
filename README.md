# fastrtc-odsc

FastRTC workshop for ODSC East 2025 

## Getting Started

1. Install requirements `pip install -r requirements.txt` or `uv pip install requirements.txt`. Make sure you do so in a virtual environment. 
2. Create a HuggingFace [account](https://huggingface.co/welcome) and generate a user access token ([Instructions](https://huggingface.co/docs/hub/en/security-tokens)). We will be using the Hugging Face API for LLM inference but you can rewrite the demos to use any LLM of your choosing.
Pro-Tip: You can create a Fine-Grained token that just has permission to call Inference Providers
![](https://github.com/user-attachments/assets/fb8ee998-4f5c-4b1b-ad22-794c04465797)
3. Place your token in the `env` file and rename the file to be `.env`
4. Optional: Add your openAI API key to OPENAI_API_KEY to `.env` to run demo 04.
4. That's it! You can now start running the demos.

## Demos

You can run any demo with `python <demo-folder-name>/app.py`. To use the built-in telephone integration, use `MODE=PHONE python <demo-folder-name>/app.py`. Not every demo is supported for phone use.

