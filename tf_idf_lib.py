import numpy as np
import math 

def term_frequency(visual_word_index : int, bovw : np.array):
    # frequency of visual word in an image 
    return bovw[visual_word_index] / np.sum(bovw)

def document_frequency(visual_word_index : int, images_histograms : np.array): 
    # in how many images visual word occur 
    df = 0 
    for _, hist in enumerate(images_histograms):
        if hist[visual_word_index] > 0:
            df += 1 
    return df 

def inverse_document_frequency(document_frequency : int, number_of_images : int) -> float:
    if document_frequency == 0:
        return 0.0
    else: 
        return math.log(number_of_images/document_frequency, 2)

def get_array_of_hists(codebook : dict) -> np.array:
    # returns array of histograms in whole dataset 
    hist_matrix = []
    for category, cat_histograms in codebook.items():
        hist_matrix.extend(np.array(cat_histograms).astype(np.int32))
    hist_matrix = np.array(hist_matrix)
    return hist_matrix 

class TF_IDF:
    def __init__(self, codebook : dict) -> None:
        self.hists = get_array_of_hists(codebook)
        self.dataset_len = self.hists.shape[0]
        self.vocab_len = self.hists.shape[1]
        self.document_frequencies = [document_frequency(vw_id, self.hists) for vw_id in range(self.vocab_len)]
        self.inverse_document_frequencies = [inverse_document_frequency(self.document_frequencies[vw_id], self.dataset_len) for vw_id in range(self.vocab_len)]

    def get_df(self, vw_id):
        return self.document_frequencies[vw_id]

    def get_idf(self, vw_id):
        return self.inverse_document_frequencies[vw_id]

    def get_tf_idf(self, vw_id, bovw): 
        tf = term_frequency(vw_id, bovw)
        idf = self.get_idf(vw_id)
        return tf*idf 



    
