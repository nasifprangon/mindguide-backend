from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import json
import pandas as pd
from dotenv import load_dotenv
import time

# Load API key from .env
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load datasets
with open("services_info.json") as f:
    SERVICE_INFO = json.load(f)

# Try to load reddit dataset
try:
    REDDIT_DF = pd.read_csv("reddit_comments.csv")
except FileNotFoundError:
    REDDIT_DF = None
    print("reddit_comments.csv not found. Reviews will be disabled.")

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    mode = data.get('mode', 'llm_csv')

    if not prompt:
        return jsonify({'reply': 'No prompt provided.', 'category': None}), 400

    try:
        if mode == 'llm_only':
            start = time.time()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            end = time.time()
            return jsonify({
                'reply': response.choices[0].message.content.strip(),
                'category': 'llm_only',
                'time': round(end - start, 2),
                'tokens': response.usage.total_tokens
            })

        system_msg = (
            "You are a classifier. Given a user's question, "
            "respond with one of the following categories:\n"
            "- info: factual questions about therapy services (e.g. pricing, insurance)\n"
            "- reviews: questions about user experiences or recommendations\n"
            "- unknown: if it doesn’t fit the above\n"
            "Only reply with one word: info, reviews, or unknown."
        )

        start = time.time()
        classification_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
        )
        classification = classification_response.choices[0].message.content.strip().lower()

        if classification == "info":
            matched = None
            for provider in SERVICE_INFO:
                if provider.lower() in prompt.lower():
                    matched = provider
                    break

            if matched:
                info = SERVICE_INFO[matched]
                info_prompt = (
                    f"The user asked: '{prompt}'\n\n"
                    f"Here is some data about the provider {matched}:\n"
                    f"{json.dumps(info, indent=2)}\n\n"
                    "Write a short summary in your own words that answers the user's query. "
                    "Avoid repeating field names and keep it friendly and informative."
                )

                gpt_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": info_prompt}]
                )

                end = time.time()
                return jsonify({
                    'reply': gpt_response.choices[0].message.content.strip(),
                    'category': 'info',
                    'time': round(end - start, 2),
                    'tokens': classification_response.usage.total_tokens + gpt_response.usage.total_tokens
                })
            else:
                end = time.time()
                return jsonify({
                    'reply': "Sorry, I couldn't find any info for that provider.",
                    'category': 'info',
                    'time': round(end - start, 2),
                    'tokens': classification_response.usage.total_tokens
                })

        elif classification == "reviews":
            if REDDIT_DF is None:
                return jsonify({
                    'reply': "Sorry, reviews are currently unavailable.",
                    'category': 'reviews',
                    'time': round(time.time() - start, 2),
                    'tokens': classification_response.usage.total_tokens
                })
            matched = None
            for provider in REDDIT_DF["website"].unique():
                if provider.lower() in prompt.lower():
                    matched = provider
                    break

            if matched:
                filtered = REDDIT_DF[REDDIT_DF["website"].str.lower() == matched.lower()]
                top_reviews = filtered.sort_values("score", ascending=False).head(5)
                prompt_text = "Summarize the following reviews:\n" + "\n".join(f"- {t}" for t in top_reviews["text"])

                review_start = time.time()
                summary_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt_text}]
                )
                review_end = time.time()

                return jsonify({
                    'reply': summary_response.choices[0].message.content.strip(),
                    'category': 'reviews',
                    'time': round(review_end - start, 2),
                    'tokens': classification_response.usage.total_tokens + summary_response.usage.total_tokens
                })
            else:
                return jsonify({
                    'reply': "Sorry, I couldn't find reviews for that provider.",
                    'category': 'reviews',
                    'time': round(time.time() - start, 2),
                    'tokens': classification_response.usage.total_tokens
                })

        elif classification == "unknown":
            fallback_prompt = (
                f"The user said: '{prompt}'\n\n"
                "You are a kind, non-judgmental AI mental health assistant. "
                "Respond empathetically in 2–3 sentences, acknowledging their emotion and encouraging them to seek professional help. "
                "Do not give clinical advice."
            )

            unknown_start = time.time()
            fallback_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": fallback_prompt}]
            )
            unknown_end = time.time()

            return jsonify({
                'reply': fallback_response.choices[0].message.content.strip(),
                'category': 'unknown',
                'time': round(unknown_end - start, 2),
                'tokens': classification_response.usage.total_tokens + fallback_response.usage.total_tokens
            })

        else:
            return jsonify({
                'reply': "Sorry, I couldn't process your request.",
                'category': 'unknown',
                'time': round(time.time() - start, 2),
                'tokens': classification_response.usage.total_tokens
            })

    except Exception as e:
        return jsonify({'reply': f"Error: {str(e)}", 'category': None, 'time': 0, 'tokens': 0}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
