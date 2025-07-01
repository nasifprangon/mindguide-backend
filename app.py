from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv

# Load API key from .env
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
        # Step 1: LLM classification (info / reviews / unknown)
        system_msg = (
            "You are a classifier. Given a user's question, "
            "respond with one of the following categories:\n"
            "- info: factual questions about therapy services (e.g. hours, contact)\n"
            "- reviews: questions about user experiences or recommendations\n"
            "- unknown: if it doesn’t fit the above\n"
            "Only reply with one word: info, reviews, or unknown."
        )

        classification_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
        )
        classification = classification_response.choices[0].message.content.strip().lower()

        # Step 2: Respond based on the classified category
        if classification == "info":
            reply = f"(info category detected)\nThis is placeholder info based on your query: '{prompt}'."

        elif classification == "reviews":
            reply = f"(reviews category detected)\nHere’s a summary from user feedback for: '{prompt}'."

        elif classification == "unknown":
            # Fallback response using GPT to handle emotional or uncategorizable input
            fallback_prompt = (
                f"The user said: '{prompt}'\n\n"
                "You are a kind, non-judgmental AI mental health assistant. "
                "Respond empathetically in 2–3 sentences, acknowledging their emotion and encouraging them to seek professional help. "
                "Do not give clinical advice."
            )

            fallback_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": fallback_prompt}]
            )
            reply = fallback_response.choices[0].message.content.strip()

        else:
            classification = "unknown"
            reply = "Sorry, I couldn't process your request."

        return jsonify({'reply': reply, 'category': classification})

    except Exception as e:
        return jsonify({'reply': f"Error: {str(e)}", 'category': None}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
