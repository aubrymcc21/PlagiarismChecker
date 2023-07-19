import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# create and clean the list of .txt files in the root folder
def load_text_notes():
    text_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
    text_notes = [open(_file, encoding='utf-8').read() for _file in text_files]
    return text_files, text_notes

# Fits & transforms the input data then convert to array
def vectorize_text(text):
    return TfidfVectorizer().fit_transform(text).toarray()

# cosine_similariy function to calculate the similarity between two docs
def calculate_similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])[0][1]

# calls the previous functions to synthesize & format the information; returns the printed results
def check_for_plagiarism():
    text_files, text_notes = load_text_notes()
    vectors = vectorize_text(text_notes)
    s_vectors = list(zip(text_files, vectors))
    plagiarism_results = set()

    for author_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((author_a, text_vector_a))
        del new_vectors[current_index]
        for author_b, text_vector_b in new_vectors:
            sim_score = calculate_similarity(text_vector_a, text_vector_b)
            sim_score = str((round(sim_score, 4) * 100)) + '%'  # Round the similarity score then multiply by 100 to convert to a percent
            author_pair = sorted((author_a, author_b))
            score = 'Similarity score between ' + str(author_pair[0]) + ' and ' + str(author_pair[1]) + ': ' + str(sim_score)
            plagiarism_results.add(score)

    return plagiarism_results

if __name__ == "__main__":
    for data in check_for_plagiarism():
        print(data)
