from chatbot.llm import ask_llm

question = input("You: ")

answer = ask_llm(question)

print("\nAI:", answer)