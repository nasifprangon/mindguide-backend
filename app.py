from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import json
import pandas as pd
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load datasets
with open("services_info.json") as f:
    SERVICE_INFO = json.load(f)

REDDIT_DF = pd.read_csv("reddit_comments.csv")

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '').strip()

    if not prompt:
        return jsonify({'reply': 'No prompt provided.', 'category': None}), 400

    try:
        # Step 1: Classify the prompt
        system_msg = (
            "You are a classifier. Given a user's question, "
            "respond with one of the following categories:\n"
            "- info: factual questions about therapy services (e.g. pricing, insurance)\n"
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

        # Step 2: Handle info category
        if classification == "info":
            matched = None
            for provider in SERVICE_INFO:
                if provider.lower() in prompt.lower():
                    matched = provider
                    break

            if matched:
                info = SERVICE_INFO[matched]
                reply = (
                    f"**{matched} - Service Info**\n"
                    f"- Website: {info.get('website', 'N/A')}\n"
                    f"- Contact: {info.get('contact', 'N/A')}\n"
                    f"- Price: {info.get('price', 'N/A')}\n"
                    f"- Platform: {info.get('platform_type', 'N/A')}\n"
                    f"- Specialties: {', '.join(info.get('specialties', []))}\n"
                    f"- Insurance: {info.get('insurance', 'N/A')}\n"
                    f"- Free Trial: {info.get('free_trial', 'N/A')}\n"
                    f"- Live Sessions: {info.get('live_sessions', 'N/A')}"
                )
            else:
                reply = "Sorry, I couldn't find any info for that provider."

        # Step 3: Handle reviews category
        elif classification == "reviews":
            matched = None
            for provider in REDDIT_DF["website"].unique():
                if provider.lower() in prompt.lower():
                    matched = provider
                    break

            if matched:
                filtered = REDDIT_DF[REDDIT_DF["website"].str.lower() == matched.lower()]
                top_reviews = filtered.sort_values("score", ascending=False).head(5)
                prompt_text = "Summarize the following reviews:\n" + "\n".join(f"- {t}" for t in top_reviews["text"])

                summary_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt_text}]
                )
                reply = summary_response.choices[0].message.content.strip()
            else:
                reply = "Sorry, I couldn't find reviews for that provider."

        # Step 4: Handle unknown category
        elif classification == "unknown":
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

        # Fallback in case something goes wrong
        else:
            classification = "unknown"
            reply = "Sorry, I couldn't process your request."

        return jsonify({'reply': reply, 'category': classification})

    except Exception as e:
        return jsonify({'reply': f"Error: {str(e)}", 'category': None}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
