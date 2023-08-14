import os
import math
import streamlit as st
from nltk.tokenize import word_tokenize
from collections import defaultdict
from transformers import pipeline

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.avg_doc_len = 0
        self.doc_lengths = defaultdict(int)
        self.inverted_index = defaultdict(lambda: defaultdict(int))

    def add_document(self, doc_id, content):
        tokens = word_tokenize(content.lower())
        self.doc_lengths[doc_id] = len(tokens)
        self.avg_doc_len += len(tokens)

        term_freq = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1

        for token, freq in term_freq.items():
            self.inverted_index[token][doc_id] = freq

    def calculate_avg_doc_len(self):
        self.avg_doc_len /= len(self.doc_lengths)

    def score(self, query):
        scores = defaultdict(float)
        query_tokens = word_tokenize(query.lower())

        for q_token in query_tokens:
            if q_token in self.inverted_index:
                doc_freq = len(self.inverted_index[q_token])
                idf = math.log((len(self.doc_lengths) - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

                for doc_id, doc_len in self.doc_lengths.items():
                    doc_freq = self.inverted_index[q_token][doc_id]
                    numerator = (self.k1 + 1) * doc_freq
                    denominator = self.k1 * ((1 - self.b) + self.b * (doc_len / self.avg_doc_len)) + doc_freq
                    scores[doc_id] += idf * (numerator / denominator)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def read_text_files_from_folder(folder_path):
    texts = []
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    text = f.read()
                    texts.append(text)
                    file_paths.append(file_path)
    return texts, file_paths

def get_most_relevant_lines(text, query):
    lines = text.split('\n')
    bm25 = BM25()
    bm25.add_document(0, text)
    bm25.calculate_avg_doc_len()
    results = bm25.score(query)
    
    if len(results) > 0:
        most_relevant_line = None
        max_count = 0
        query_tokens = word_tokenize(query.lower())
        for idx, line in enumerate(lines):
            line_tokens = word_tokenize(line.lower())
            count = sum([1 for token in query_tokens if token in line_tokens])
            if count > max_count:
                max_count = count
                most_relevant_line = idx

        if most_relevant_line is not None:
            start_line = max(0, most_relevant_line - 1)
            end_line = min(len(lines), most_relevant_line + 11)  # Extract the next 10 lines after the most relevant line
            return lines[start_line:end_line]
    
    return None

@st.cache(allow_output_mutation=True)
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@st.cache(allow_output_mutation=True)
def load_documents(folder_path):
    texts, file_paths = read_text_files_from_folder(folder_path)

    bm25 = BM25()
    for idx, text in enumerate(texts):
        bm25.add_document(idx, text)
    bm25.calculate_avg_doc_len()

    return bm25, texts

def process_query(bm25, texts, qa_pipeline, query):
    results = bm25.score(query)

    if len(results) > 0:
        most_relevant_doc_id, _ = results[0]
        most_relevant_text = texts[most_relevant_doc_id]
        relevant_lines = get_most_relevant_lines(most_relevant_text, query)

        if relevant_lines is not None:
            context = " ".join(relevant_lines)  # Combine the relevant lines into a single string as context
            question = query

            # Get the precise answer from the context using the question-answering model
            answer = qa_pipeline(question=question, context=context)

            return relevant_lines, answer
    return None, None

def main():
    folder_path = "./"  # Replace this with the path to your folder containing text files

    # Load the QA model and documents
    qa_pipeline = load_qa_model()
    bm25, texts = load_documents(folder_path)

    st.title("Question Answering System")
    
    query = st.text_input("Ask a question:")
    
    if query:
        relevant_lines, answer = process_query(bm25, texts, qa_pipeline, query)
        
        if relevant_lines is not None and answer is not None:
            st.write("Most relevant lines:")
            for line in relevant_lines:
                st.write(line)

            st.write("\nPrecise Answer:")
            st.write(answer["answer"])
        else:
            st.write("No relevant text or lines found.")

if __name__ == "__main__":
    main()

