from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_groq import ChatGroq
import uvicorn

# Load .env file (containing your GROQ_API_KEY)
load_dotenv()

# Initialize FastAPI app and Groq model
app = FastAPI()
llm = ChatGroq(model="openai/gpt-oss-120b")

# Chat history kept in memory on your laptop
messages = []


@app.get("/chat/{user_input}")
def chat_with_ai(user_input: str):
    # 1. Save user question to memory
    messages.append(("user", user_input))

    # 2. Send full history to Groq AI
    response = llm.invoke(messages)

    # 3. Save AI response to memory
    messages.append(("assistant", response.content))

    # 4. Send AI answer back to your phone
    return {"question": user_input, "ai_response": response.content}


if __name__ == "__main__":
    # Host 0.0.0.0 allows your phone on the same Wi-Fi to connect
    uvicorn.run(app, host="0.0.0.0", port=8000)