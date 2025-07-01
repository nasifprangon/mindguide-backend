from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '').strip()

    if not prompt:
        return jsonify({'reply': 'No prompt provided.', 'category': None}), 400

    try:
        # Step 1: Classify the prompt into a category
        system_msg = (
            "You are a classifier. Given a user's question, "
            "respond with just one word: either 'info' or 'reviews'."
        )
        classification_response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can switch to gpt-3.5-turbo if needed
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
        )
        classification = classification_response.choices[0].message.content.strip().lower()

        # Step 2: Respond based on the classified category
        if classification == "info":
            # TODO: Replace with actual website info lookup
            reply = f"(info category detected)\nThis is placeholder info based on your query: '{prompt}'."
        elif classification == "reviews":
            # TODO: Replace with actual Reddit review summarization
            reply = f"(reviews category detected)\nHere’s a summary from user feedback for: '{prompt}'."
        else:
            classification = "unknown"
            reply = "Sorry, I couldn't classify your question clearly. Please try again."

        return jsonify({'reply': reply, 'category': classification})

    except Exception as e:
        return jsonify({'reply': f"Error: {str(e)}", 'category': None}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
