from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
    return _model, _tokenizer


def generate_response(prompt: str) -> str:
    """사용자 입력(prompt)을 넣으면 LLM 응답 한 문단을 돌려준다."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are Qwen, created by Alibaba Cloud. "
                "You are a helpful assistant. "
                "You speak Korean all the time."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated = model.generate(**model_inputs, max_new_tokens=512)
    # 프롬프트 토큰 잘라내기
    generated = generated[:, model_inputs.input_ids.shape[-1] :]

    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
