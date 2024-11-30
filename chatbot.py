import re

class GeneralChatbot:
    def __init__(self):
        self.rules = {
            "greeting": self.handle_greeting,
            "ask_help": self.handle_ask_help,
            "farewell": self.handle_farewell,
            "default": self.handle_default,
        }

    def respond(self, user_input):
        user_input = user_input.lower()
        intents = {
            "greeting": r"(hello|hi|hey|greetings)",
            "ask_help": r"(help|assist|support)",
            "farewell": r"(bye|goodbye|see you|farewell)",
        }

        matched_intent = None
        for intent, pattern in intents.items():
            if re.search(pattern, user_input):
                matched_intent = intent
                break

        if matched_intent in self.rules:
            return self.rules[matched_intent](user_input)
        else:
            return self.rules["default"](user_input)

    def handle_greeting(self, user_input):
        return "Hello! How can I assist you today?"

    def handle_ask_help(self, user_input):
        return "I am here to help. Please let me know what you need assistance with."

    def handle_farewell(self, user_input):
        return "Goodbye! Have a great day!"

    def handle_default(self, user_input):
        return f"I didn't understand that. Could you please rephrase your question? Your input: `{user_input}`"

def main():
    chatbot = GeneralChatbot()
    print("Welcome to the General Chatbot. How can I assist you today? (Type 'quit' to exit)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Thank you for using the General Chatbot. Goodbye!")
            break
        print("Chatbot:", chatbot.respond(user_input), "\n")

if __name__ == "__main__":
    main()
