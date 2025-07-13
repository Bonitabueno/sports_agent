from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "당신의 이름은 무엇인가요?"
messages = [
    {"role": "system", "content": (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        "You speak Korean all the time."
      )
    },
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
st.write(response)


import streamlit as st

# 챗봇 인터페이스 메인 함수
# 스피너 없음
def main():
    st.title("스포츠 LLM")
    st.write("스포츠 정보 제공 에이전트 도전")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 메시지 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if user_input := st.chat_input("질문을 입력하세요:"):
        # 사용자 메시지 출력
        st.chat_message("user").markdown(user_input)
        # 메시지 히스토리에 추가
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 사용자 입력 처리
        response = process_input(user_input)  # process_input 함수 호출
        assistant_response = response

        # 챗봇 응답 출력
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        # 메시지 히스토리에 추가
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    main()
