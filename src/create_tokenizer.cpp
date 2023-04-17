//
// Created by Erik Hyrkas on 4/16/2023.
//
#include <iostream>
#include "ml\byte_pair_encoder.hpp"


using namespace happyml;

// This is a simple program that will create a tokenizer from a single text file or folder of text files.
// Here are some suggestions on sources of text:
//
// * Wikipedia: Wikipedia is a vast source of text data in multiple languages. You can download the entire Wikipedia corpus or a subset of it for specific languages. Wikipedia is a good choice for training a BPE model because it contains a diverse range of topics and writing styles.
// * OpenWebText: OpenWebText is a dataset of web pages that has been filtered and cleaned to remove low-quality content. It is available in multiple languages and is a good choice for training a BPE model because it contains a diverse range of text from the internet.
// * News Crawl: The News Crawl dataset is a collection of news articles from multiple sources and languages. It is a good choice for training a BPE model if you are interested in working with news text specifically.
// * Books Corpus: The Books Corpus is a collection of over 11,000 books in multiple genres and languages. It is a good choice for training a BPE model if you are interested in working with literary text specifically.
//
// Training on a large amount of data takes time. The single text file approach will take time to scan to
// build the vocabulary, where processing a whole folder with many files will be less accurate but faster.
//
// On Windows, I ran the following command to combine all the text files in a folder:
//   copy .\text\* .\internet.txt
//
// On Linux, You could run the following command to combine all the text files in a folder:
//   cat ./text/* > ./internet.txt
//
// USAGE: creat_tokenizer path
// path: path to a single text file or folder of text files
//
// Example with file:  .\create_tokenizer.exe ..\data\internet.txt
// Example with folder:  .\create_tokenizer.exe ..\data\text
//
// The tokenizer will be saved to: ../happyml_repo/default_token_encoder
int main(int argc, char *argv[]) {
    try {
        if( argc < 2 ) {
            cout << "USAGE: creat_tokenizer path" << endl;
            return 1;
        }
        string path = argv[1];
        if( path.empty() ) {
            cout << "USAGE: creat_tokenizer path" << endl;
            return 1;
        }

        BytePairEncoderModel bpe;
        if(filesystem::is_directory(path) ) {
            bpe.train_on_folder(path);
        } else if(filesystem::is_regular_file(path) ) {
            bpe.train_on_file(path);
        } else {
            cout << "USAGE: creat_tokenizer path" << endl;
            return 1;
        }
        bpe.save("../happyml_repo", "default_token_encoder", true);
    } catch (const exception &e) {
        cout << e.what() << endl;
        return 1;
    }
    return 0;
}