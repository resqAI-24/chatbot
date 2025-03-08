from model import Chatbot

def main():
    bot = Chatbot("dataset/emergency_data.json")
    print("🔴 ResQAI Emergency Chatbot 🔴\n(Type 'exit' to stop)")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye! Stay safe. 🚀")
            break

        response = bot.get_response(user_input)
        print(f"ResQAI: {response}")

if __name__ == "__main__":
    main()
