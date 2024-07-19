#include "nmt_model.hpp"

int main() {
    NMTModel model;

    // Preprocess the data
    model.process_text_data("/content/deu.txt", "english-german.pkl");

    // Load and prepare dataset
    auto dataset = model.load_dataset("english-german.pkl");
    // Split dataset into train and test sets
    size_t n_sentences = 20000;
    auto [train, test] = model.split_dataset(dataset, n_sentences);

    // Prepare tokenizers and sequences
    auto [english_tokenizer, english_vocab_size, english_max_len] = model.prepare_tokenizer(train.first);
    auto [german_tokenizer, german_vocab_size, german_max_len] = model.prepare_tokenizer(train.second);

    auto training_sequences_X = model.prepare_sequence_encoding(german_tokenizer, german_max_len, train.second);
    auto training_sequences_Y = model.prepare_sequence_encoding(english_tokenizer, english_max_len, train.first);
    auto testing_sequences_X = model.prepare_sequence_encoding(german_tokenizer, german_max_len, test.second);
    auto testing_sequences_Y = model.prepare_sequence_encoding(english_tokenizer, english_max_len, test.first);

    // Train the model
    SessionOptions session_options;
    unique_ptr<Session> session(NewSession(session_options));
    model.train_model(session.get(), training_sequences_X, training_sequences_Y, testing_sequences_X, testing_sequences_Y);

    // Evaluate the model
    auto trained_model = model.load_model("model.meta");
    model.evaluate_model(trained_model, testing_sequences_X, testing_sequences_Y, test);

    // Custom testing
    string input_sentence = "gib mir etwas";
    auto translated_sentence = model.translate_sentence(trained_model, input_sentence, german_tokenizer, german_max_len, english_tokenizer);
    cout << "Translated sentence: " << translated_sentence << endl;

    return 0;
}
