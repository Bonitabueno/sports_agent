from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 모델·토크나이저는 한 번만 로드
model = AutoModelForCausalLM.from_pretrained(
    _MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

def generate_response(messages: list[dict]) -> str:
    system_msg = {
        "role": "system",
        "content": (
            "You are Sportato."
            "You are a Sports Information Expert."
            "You speak Korean all the time."
        ),
    }

    chat = [system_msg] + messages  # 히스토리 앞에 시스템 메시지 추가

    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=512)
    output = output[:, inputs.input_ids.shape[-1] :]

    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]
