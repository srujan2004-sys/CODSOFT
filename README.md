 CHATBOT-WITH-RULE-BASED-RESPONSES

- This Python code defines a simple rule-based chatbot that responds to user inputs based on predefined patterns. Here's how it 
  works:

- The rule_based_chatbot function takes an input text from the user and matches it against a set of regular expression patterns 
  defined in the rules dictionary.

- Each key in the rules dictionary represents a regular expression pattern, and the corresponding value is the response the 
  chatbot should give if the input matches that pattern.

- If the input text matches any of the patterns, the chatbot responds with the corresponding value from the rules dictionary.

- If none of the patterns match the input text, the chatbot defaults to a generic response.

- The while True loop continuously prompts the user for input until the user types "exit", at which point the chatbot prints a 
  goodbye message and breaks out of the loop, ending the conversation.

- Inside the loop, the user input is processed by the rule_based_chatbot function, and the bot's response is printed.


Tic-Tac-Toe AI
A simple implementation of the classic Tic-Tac-Toe game with an AI opponent. This project allows two players to play against each other or against a computer AI that uses the Minimax Algorithm to make optimal moves.

