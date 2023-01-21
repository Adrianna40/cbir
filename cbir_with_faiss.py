# %%
import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from typing import List, T
from sklearn.model_selection import train_test_split
import faiss
import time

# %%
project_config = {
    'dataset_path' : 'caltech-101/101_ObjectCategories',
    'train_size' : 20,
    'test_size' : 4,
    'k' : 500,
    'number_of_categories' : 20
}

# %%
# finding categories that have enough images 
dataset_size = project_config['train_size'] + project_config['test_size']
categories = []
for root, dirs, files in os.walk(project_config['dataset_path']):
    for dir in dirs:
        for cat_root, cat_dirs, cat_files in os.walk(os.path.join(project_config['dataset_path'], dir)):
            if len(cat_files) >= dataset_size:
                categories.append(dir)

# %%
def get_file_paths(dataset_path : str, category : str, test_size : float) -> List[str]:
    """ 
    returns names of files for test and train datasets of particular category
    """
    files_paths = []
    for root, dirs, files in os.walk(os.path.join(dataset_path, category)):
        for name in files:
            files_paths.append(os.path.join(root, name))
    train_files, test_files = train_test_split(files_paths, test_size=test_size)
    return train_files, test_files

def load_images(files_paths : List[str]) -> List[np.ndarray]:
    images = []
    for file in files_paths:
        img = cv2.imread(file,0)
        images.append(img)
    return images
        

# %%
# creating train and test datasets 
train_images = {}
test_images = {}
for cat in categories[:project_config['number_of_categories']]:
    train_files, test_files = get_file_paths(project_config['dataset_path'], cat, project_config['test_size'])
    train_images[cat] = load_images(train_files)
    test_images[cat] = load_images(test_files)

# %%
def sift_features(images : List) -> List:
    descriptor_dict = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for category, cat_images in images.items():
        features = []
        for img in cat_images:
            kp, des = sift.detectAndCompute(img,None)
            if (des is not None):
                descriptor_list.extend(des)
                features.append(des)
            else: 
                print(f'image number {cat_images.index(img)} has no descriptors')
        descriptor_dict[category] = features
    return [descriptor_list, descriptor_dict]

print('train')
descriptor_list, features_by_category_train = sift_features(train_images) 
print('test')
_, features_by_category_test = sift_features(test_images)
# descriptor_list is needed for creating clustering centers, so we only take them from train dataset
# bovw_feature for train and test are needed for classification of the image 

# %%
# Returns an array that holds central points (visual words)
def kmeans(k : int, descriptor_list : List) -> List:
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    return visual_words
"""
start = time.time()
visual_words = kmeans(project_config['k'], descriptor_list) 
end = time.time()
print('sklearn kmeans', end-start)
"""

# much slower than faiss
# comparison of time : 
# sklearn kmeans 179.26710486412048
# faiss kmeans 0.9870326519012451

# %%

# kernel crashes
train_data = np.array(descriptor_list)
start = time.time()
kmeans_faiss = faiss.Kmeans(d=train_data.shape[1], k=project_config['k'])
kmeans_faiss.train(descriptor_list)
end = time.time()
print('faiss kmeans', end-start)


# %%
def cosine_similarity(vec1 : np.ndarray, vec2 : np.ndarray) -> float:
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

# %%
def find_index(feature : np.ndarray, centers : np.ndarray):
    # returns an index of the visual word that is the most similar to a feature 
    minimum = 0
    start_min = True 
    min_index = 0 
    for idx, center in enumerate(centers): 
        # we are using numpy functions, because it is faster 
        distance_c = cosine_similarity(center, feature)
        if start_min:
            minimum = distance_c
            start_min = False 
        else: 
            minimum = min(minimum, distance_c)
            if minimum == distance_c:
                min_index = idx

    return min_index

# %%

def generate_codebook(features_by_category : dict, kmeans) -> dict:
    """ 
    Takes 2 parameters: 
    - a dictionary, key: category name, value: list of np.array of desciptors per image 
    - an array that holds the central points (visual words) of the k means clustering
    Returns a dictionary, key: category name, value: list of bags of visual words
    """
    bovw_per_category = {}
    for category, features in features_by_category.items():
        #print(category)
        category_bovw = []
        for img_features in features:
            # create bovw for an image
            image_bovw = np.zeros(project_config['k'])
            for feature in img_features:
                # ind = find_index(feature, centers) # classification of a feature to a visual word 
                ind = kmeans.index.search(feature, 1)[1]
                image_bovw[ind] += 1
            category_bovw.append(image_bovw)
        bovw_per_category[category] = category_bovw
    return bovw_per_category
    

#codebook_train = generate_codebook(features_by_category_train, visual_words) 
codebook_train = generate_codebook(features_by_category_train, kmeans_faiss) 

#codebook_test = generate_codebook(features_by_category_test, visual_words) 
codebook_test = generate_codebook(features_by_category_test, kmeans_faiss) 

# %%
import json

def save_to_json(file_name : str, bovw : dict) -> None:
    bovw_list = {cat : list(bovw[cat][0]) for cat in bovw.keys()}
    json_object = json.dumps(bovw_list)
    with open(file_name, 'w') as outfile:
        outfile.write(json_object)

save_to_json('codebook_test.json', codebook_test)
save_to_json('codebook_train.json', codebook_train)
 

# %%
def hist_to_tfidf(codebook : dict, tf_idf : T) -> dict:
    tf_idfs_per_category = {}
    for category, list_of_hists in codebook.items():
        tf_idfs = []
        for bovw in list_of_hists:
            tf_idf_vector = [tf_idf.get_tf_idf(vw_id, bovw) for vw_id in range(bovw.shape[0])]
            tf_idfs.append(np.array(tf_idf_vector))
        tf_idfs_per_category[category] = tf_idfs
    return tf_idfs_per_category


# %%
import tf_idf_lib 
tf_idf = tf_idf_lib.TF_IDF(codebook_train)
tf_idfs_per_cat_train = hist_to_tfidf(codebook_train, tf_idf)
tf_idfs_per_cat_test = hist_to_tfidf(codebook_test, tf_idf)

# %%
# Credit: this code was a part of exercise from Scientific Programming, it was only modified by us 

class OrderedListTuple:
    # each element should be tuple with category name and cosine similarity 
    def __init__(self, max_size):
        self.content = []
        self.max_size = max_size
        
    def find_pos (self, element):
        index = 0
        while (index <= len(self.content)-1) and self.content[index][1] > element[1]:
            index += 1
        return index

    def insert_element (self, element):
        pos = self.find_pos (element)
        self.content.insert (pos, element)
        if len(self.content) > self.max_size:
            self.content.pop()

# %%
def get_top_similar_images(image_vector :np.array, train_vectors_dict : dict, max_top : int, similarity_func) -> T:
    similar_images = OrderedListTuple(max_top)
    for category, category_vectors in train_vectors_dict.items():
        for idx, train_vec in enumerate(category_vectors):
            similarity = similarity_func(image_vector, train_vec)
            similar_images.insert_element(((category, idx), similarity))
    return similar_images

# %%
test_image = tf_idfs_per_cat_test['butterfly'][0]
si = get_top_similar_images(test_image, tf_idfs_per_cat_train, 3, cosine_similarity)

# %%
si.content

# %%
from matplotlib import pyplot as plt 
fig, ax = plt.subplots(2)
pred_cat = si.content[0][0][0]
pred_id = si.content[0][0][1]
fig.tight_layout(pad=3.0)
ax[1].imshow(train_images[pred_cat][pred_id], cmap='gray')
ax[1].set_title('The most similar image from train set')
ax[0].imshow(test_images['butterfly'][0], cmap='gray')
ax[0].set_title('Image from test set')

# %%
def test_image_retrieval(test_vectors_dict, train_vectors_dict, similarity_function, max_in_top):

    reciprocal_rank_sum = 0
    is_in_top3 = 0 
    test_data_len = 0
    results = {}
    for category, vectors in test_vectors_dict.items():
        preds = []
        for test_vec in vectors:
            test_data_len += 1
            top_similar = get_top_similar_images(test_vec, train_vectors_dict, max_in_top, similarity_function)
            category_predictions = [element[0][0] for element in top_similar.content]
            preds.extend(category_predictions)
            if category in category_predictions:
                if category in category_predictions[:3]:
                    is_in_top3 += 1 
                rank = category_predictions.index(category) + 1 
                reciprocal_rank_sum += 1/rank
        results[category] = preds

    mean_reciprocal_rank = reciprocal_rank_sum / test_data_len
    is_in_top3_perc = is_in_top3 / test_data_len 
    print('Results for image retrieval')
    print('Mean reciprocal rank', mean_reciprocal_rank)
    print('How often (in per cent) the correct category is in top3', is_in_top3_perc*100, '%')
    return results, mean_reciprocal_rank, is_in_top3_perc


# %%
print('tf-idf with cosine similarity')
results, mean_reciprocal_rank, is_in_top = test_image_retrieval(tf_idfs_per_cat_test, tf_idfs_per_cat_train, cosine_similarity, 5)


# %%
print('histograms with cosine similarity')
results, mean_reciprocal_rank, is_in_top = test_image_retrieval(codebook_test, codebook_train, cosine_similarity, 5)

# %%
def common_words(hist1, hist2):
    # count every co-occurance of visual word in both images 
    common_words = 0
    for vw_id in range(len(hist1)):
        if hist1[vw_id] > 0 and hist2[vw_id] > 0:
            common_words += min(hist1[vw_id], hist2[vw_id])
    return common_words


# %%
print('histograms with common words')
results, mean_reciprocal_rank, is_in_top = test_image_retrieval(codebook_test, codebook_train, common_words, 5)

# %%
def euclidean_similarity(vec1, vec2):
    return -distance.euclidean(vec1, vec2)

# %%
print('tf-idf with euclidean similarity')
results, mean_reciprocal_rank, is_in_top = test_image_retrieval(tf_idfs_per_cat_test, tf_idfs_per_cat_train, euclidean_similarity, 5)

# %%
def braycurtis_similarity(vec1, vec2):
    return -distance.braycurtis(vec1, vec2)
print('tf-idf with braycurtis similarity')
results, mean_reciprocal_rank, is_in_top = test_image_retrieval(tf_idfs_per_cat_test, tf_idfs_per_cat_train, braycurtis_similarity, 5)

# %%



