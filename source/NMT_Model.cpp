//
// Created by John on 7/17/2024.
//

#include "NMT_Model.hpp"

NMTModel::NMTModel() {
    // Initialize default values or load from configuration
    source_vocab_size = 10000; // Example value
    target_vocab_size = 10000; // Example value
    num_units = 256;
    batch_size = 64;
    target_time_steps = 50; // Example value
}

NMTModel::~NMTModel() {
    // Cleanup if needed
}

string NMTModel::load_doc(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Failed to open " << filename << endl;
        return "";
    }
    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

vector<vector<string>> NMTModel::to_pairs(const string& doc) {
    vector<vector<string>> pairs;
    stringstream ss(doc);
    string line;
    while (getline(ss, line)) {
        vector<string> pair;
        stringstream ls(line);
        string token;
        while (getline(ls, token, '\t')) {
            pair.push_back(token);
        }
        pairs.push_back(pair);
    }
    return pairs;
}

string NMTModel::normalize_text(const string& line, const regex& re_print, const unordered_map<char, char>& table) {
    string result;
    for (char c : line) {
        if (table.find(c) != table.end()) continue;
        if (regex_match(string(1, c), re_print)) continue;
        result.push_back(tolower(c));
    }
    return result;
}

vector<vector<string>> NMTModel::clean_lines(const vector<vector<string>>& lines) {
    vector<vector<string>> cleaned;
    regex re_print("[^ -~]");
    unordered_map<char, char> table;
    for (char c : string("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")) {
        table[c] = c;
    }

    for (const auto& pair : lines) {
        vector<string> clean_pair;
        for (const auto& line : pair) {
            clean_pair.push_back(normalize_text(line, re_print, table));
        }
        cleaned.push_back(clean_pair);
    }
    return cleaned;
}

void NMTModel::save_clean_data(const vector<vector<string>>& sentences, const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        cout << "Failed to save " << filename << endl;
        return;
    }
    for (const auto& pair : sentences) {
        file << pair[0] << "\t" << pair[1] << "\n";
    }
    cout << "Saved: " << filename << endl;
}

void NMTModel::process_text_data(const string& filename, const string& output_filename) {
    string doc = load_doc(filename);
    if (!doc.empty()) {
        auto pairs = to_pairs(doc);
        auto clean_pairs = clean_lines(pairs);
        save_clean_data(clean_pairs, output_filename);
        // Spot check
        for (size_t i = 0; i < 100 && i < clean_pairs.size(); ++i) {
            cout << "[" << clean_pairs[i][0] << "] => [" << clean_pairs[i][1] << "]" << endl;
        }
    }
}

void NMTModel::train_model(Session* session, const vector<vector<int>>& training_sequences_X, const vector<vector<int>>& training_sequences_Y, const vector<vector<int>>& testing_sequences_X, const vector<vector<int>>& testing_sequences_Y) {
    // Define the model architecture using TensorFlow C++ API
    // This part requires translating the Keras model definition to TensorFlow C++ API
    // Here is a simplified example, actual implementation may vary

    // Placeholder for inputs
    Placeholder input_placeholder("input", DT_FLOAT);
    Placeholder target_placeholder("target", DT_FLOAT);

    // Define the model architecture
    // Embedding layer
    Variable embedding("embedding", DT_FLOAT, TensorShape({source_vocab_size, num_units}));
    auto embed = EmbeddingLookup(embedding, input_placeholder);

    // LSTM layers
    LSTM lstm1(num_units);
    auto lstm_output1 = lstm1(embed);

    RepeatVector repeat_vector(target_time_steps);
    auto repeated_output = repeat_vector(lstm_output1);

    LSTM lstm2(num_units, true);
    auto lstm_output2 = lstm2(repeated_output);

    TimeDistributed dense(Dense(target_vocab_size, "softmax"));
    auto output = dense(lstm_output2);

    // Loss and optimizer
    auto loss = Mean(SparseCategoricalCrossentropy(target_placeholder, output));
    Adam optimizer;
    auto train_op = optimizer.Minimize(loss);

    // Initialize variables
    session->Run({}, {}, {"init"}, nullptr);

    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        // Batch training
        for (size_t i = 0; i < training_sequences_X.size(); i += batch_size) {
            // Prepare batch data
            auto batch_X = training_sequences_X[i];
            auto batch_Y = training_sequences_Y[i];
            session->Run({{input_placeholder, batch_X}, {target_placeholder, batch_Y}}, {train_op}, nullptr);
        }
        // Validation and early stopping logic can be implemented here
    }

    // Save the trained model
    MetaGraphDef meta_graph_def;
    session->ExportMetaGraph(&meta_graph_def);
    WriteBinaryProto(Env::Default(), "model.meta", meta_graph_def);
}
