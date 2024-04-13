from flask import Flask, render_template, request
from flask_wtf import CSRFProtect
from transformers import pipeline

app = Flask(__name__)
app.secret_key = b'_53oi3uriq9pifpff;apl'
csrf = CSRFProtect(app)


conversation_history_BERT = []

model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

def read_text_file(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        return file.read()

@app.route('/ask-bert', methods=['POST'])
def ask_bert():
    if request.method == 'POST':
        input_text = request.form['input']
        conversation_history_BERT.append(["You", input_text])

        context = read_text_file("./dsa_questions.txt")

        QA_input = {"question": str(input_text),
                    "context": context}
        response = nlp(QA_input)
        answer = response.get('answer', 'No answer found')
        conversation_history_BERT.append(["Bot", answer])

        return render_template('bert.html', value = conversation_history_BERT)


@app.route('/bert')
def bert():
    return render_template('bert.html', value = conversation_history_BERT)



'''
def train_ebook_model_GPT():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model_config = GPT2Config.from_pretrained("gpt2")

    model = GPT2LMHeadModel(model_config)

    def load_ebook(filename):
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()

    ebook_data = load_ebook("./dsa.txt")  # Assuming the e-book file is named "dsa.txt"

    input_ids = tokenizer.encode(ebook_data, return_tensors="tf", truncation=True, max_length=512)

    batch_size = 4
    epochs = 3

    optimizer = Adam(learning_rate=5e-5)
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i:i + batch_size]
            with tf.GradientTape() as tape:
                outputs = model(batch_input_ids)
                logits = outputs.logits[:, :-1, :]  # Remove the last token from logits
                labels = batch_input_ids[:, 1:]  # Shift input ids to the right for labels
                loss = loss_fn(labels, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print(f"Batch {i // batch_size + 1}/{len(input_ids) // batch_size} - Loss: {loss.numpy():.4f}")

    model.save_pretrained("./gpt2-finetuned")

    return model, tokenizer


def generate_response_GPT(input_text, model, tokenizer, max_length=100):
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    response = generator(input_text, max_length=max_length, num_return_sequences=1)
    return response[0]["generated_text"]


@app.route('/ask-gpt', methods=['POST'])
def ask_gpt():
    if request.method == 'POST':
        input_text = request.form['input']
        conversation_history_GPT.append(["You", input_text])

        model, tokenizer = train_ebook_model_GPT()

        output_text = generate_response_GPT(input_text, model, tokenizer)
        conversation_history_GPT.append(["Bot", output_text])

        return render_template('gpt.html', value=conversation_history_GPT)

@app.route('/gpt')
def gpt():
    return render_template('gpt.html', value= conversation_history_GPT)



def load_intents(filename):
    with open(filename, "r", encoding="utf-8") as file:
        intents = json.load(file)
    return intents

def get_response(input_text, intents):   # O(n)
    for intent in intents["intents"]:
        pattern = ' '.join([str(s) for s in intent['patterns']])
        if input_text.lower() in pattern.lower():
            response = ' '.join([str(s) for s in intent['responses']])
            return response

    return "Sorry, I didn't understand that."

@app.route('/ask-chatbot', methods=['POST'])
def ask_chatbot():
    if request.method == 'POST':
        input_text = request.form['input']
        conversation_history_chatbot.append(["You", input_text])

        intents = load_intents("data.json")

        output_text = get_response(input_text, intents)
        conversation_history_chatbot.append(["Bot", output_text])

        return render_template('chatbot.html', value = conversation_history_chatbot)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', value= conversation_history_chatbot)

'''
@app.route('/')
def home():
    return render_template('home.html', value='')



if __name__ == '__main__':
    app.run(debug=True)
