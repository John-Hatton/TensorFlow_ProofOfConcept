//
// Created by John on 7/17/2024.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <regex>
#include <algorithm>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
//#include <tensorflow/core/protobuf/meta_graph.proto>

#ifndef TENSORFLOW_PROOFOFCONCEPT_NMT_MODEL_HPP
#define TENSORFLOW_PROOFOFCONCEPT_NMT_MODEL_HPP

using namespace tensorflow;
using namespace std;

class NMTModel {
public:
    NMTModel();
    ~NMTModel();

    // Utility functions for file handling and text preprocessing
    string load_doc(const string& filename);
    vector<vector<string>> to_pairs(const string& doc);
    string normalize_text(const string& line, const regex& re_print, const unordered_map<char, char>& table);
    vector<vector<string>> clean_lines(const vector<vector<string>>& lines);
    void save_clean_data(const vector<vector<string>>& sentences, const string& filename);
    void process_text_data(const string& filename, const string& output_filename);

    // Model training and evaluation using TensorFlow C++
    void train_model(Session* session, const vector<vector<int>>& training_sequences_X, const vector<vector<int>>& training_sequences_Y, const vector<vector<int>>& testing_sequences_X, const vector<vector<int>>& testing_sequences_Y);

private:
    size_t source_vocab_size;
    size_t target_vocab_size;
    size_t num_units;
    size_t batch_size;
    size_t target_time_steps;
};

#endif //TENSORFLOW_PROOFOFCONCEPT_NMT_MODEL_HPP
