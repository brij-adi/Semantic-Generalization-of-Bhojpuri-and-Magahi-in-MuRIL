# this line helps in importing any transformer model and tokenizer without specifying it
#cause if we didnt do that we would have to import each model and its tokenizer separately
#this is a generic approach and the most feasible one
#finally we need not need to write code for every architecture separately
from transformers import AutoTokenizer, AutoModel 

#imports pytorch it runs the neural network computations for muril
#since its approach is based on pytorch
import torch

#so basically the thing is that the embeddings generated are in the form of tensors
#and even though tensors are similar to numpy arrays they cannot be understood by sklearn since tensors run on gpu and sklearn doesnt understand gpu data
#hence the conversion is necessary
#plus sklearn provides us with a working function to calculate cosine similarity instead of writing it manually
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#PCA= Principal Component Analysis
#it reduces the number of dimensions of the embeddings so that its easier to be visualized by TSNE
#this introduces speed, smoother clustering and less noise in the data
from sklearn.decomposition import PCA

#TSNE = T-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING
#a visualization algorithm
#projects high dimensional data like 768D to 2D or 3D so that humans can see patterns
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

#a string that stores the model name
MODEL = "google/muril-base-cased"

#loads the tokenizer that was used by muril
#ensures that wordpieces subwords and vocabularies and token IDs match what muril expects
tokenizer = AutoTokenizer.from_pretrained(MODEL)


#loads the model muril model from huggingface
#equivalent to loading the neural network weights trained by google for multilingual tasks
model = AutoModel.from_pretrained(MODEL)

#models in pytorch have 2 modes
# training and evaluation
#the latter is used as we are not training the model but just using it to generate embeddings
model.eval()

#a function , it takes a sentence as the input
#returns a CLS or vector representation of the sentence using the model's CLS token
def get_cls_embedding(sentence):
    #calls the tokenizer converts the sentence into numbers that the model can understand
    #sentence= text that is to be encoded, return_tensors= returns pytorch tensors instead of lists, truncation= true means if the sentence is too long cut it to the max lenght allowed, 
    #padding= true ensures that the input is padded so that sequence length is uniform
    #result inputs is a dictionary of tensors
    inputs = tokenizer(sentence,return_tensors="pt",truncation=True,padding=True)
    #tells pytorch to not calculate gradients as we are not training only inferencing
    #saves memory and improves speed
    with torch.no_grad():
        #calls the model on the tokenized inputs
        #inputs unpacks the dictionary
        out = model(**inputs)
    #embeddings for every token
    #sometimes more field depending on the model
    cls = out.last_hidden_state[:,0,:]
    #cls.cpu() moves the tensor from gpu to cpu
    #.numpy() convets pytorch tensor to numpy array
    #.reshape(-1) flattens the array to 1D vector
    return cls.cpu().numpy().reshape(-1)

#argument is a list of sentences
#embs gets the CLS embeddings for each sentence in the list by looping over each sentence
#get_cls_embeddings(s) returns a 768D embedding vector for each sentence
#stores all those vectors in a python list named embs
#so if 3 sentences passed we get 3 vectors each of shape 768D
#np,vstack() takes the list of 1-D embeddings and stacks them into 2-D numpy array
#so the function returns one matrix containing all sentence embeddings
def batch_embeddings(sentences):
    embs = [get_cls_embedding(s) for s in sentences]
    return np.vstack(embs)


#a function that takes 2 sets of 2-D matrix embeddings
#each should have a shape of(N,768) and (M,768) where N and M are number of sentences
#cosine similarity from sklearn computes the similarity between every row in a and every row in b
#output is a matrix of shape(N,M) where each cell contains the cosine similarity between the sentences
def pairwise_sim(a,b):
    return cosine_similarity(a,b)

#a python dictionary containing sentences in Hindi,Bhojpuri and Magahi
#all languages are semantically aligned but linguistically different
#this dictionary is later used to extract sentences and compute their embeddings
sent_simple = {
    "Hindi": "आज मेरा जन्मदिन है,परंतु किसीने मुझे बधाई नहीं दी।",
    "Bhojpuri": "आजु हमार जनमदिन ह, लेकिन केहू हमरा के बधाई ना दिहल।",
    "Magahi": "आज हमर जनमदिन हे, लेकिन कोई हमरा बधाई न देलक।"
}


#loops over each key-value pair in the dictionary
#lang is the language name and s is the sentence
for lang, s in sent_simple.items():
    #tokenized the sentence s
    #converts it into pytorch tensors
    #if sentence is too long truncates it to max length
    #enc is a dictionary containing input ids and attention masks
    #attention masks are automatically generated by the tokenizer
    #used to tell the model which tokens are real and which are padding
    enc = tokenizer(s, return_tensors="pt", truncation=True)
    #extract the first and only sequence of input IDs
    #converts from torch tensor to python lists
    ids = enc["input_ids"][0].tolist()
    #converts each intenger token ID bacn into its string token
    tokens = tokenizer.convert_ids_to_tokens(ids)
    #prints the language as a header
    print(f"\n--- {lang} ---")
    #prints the token ids for the sentence
    print("input_ids:", ids)
    #prints the list of tokens for the sentence
    print("tokens   :", tokens)
    #checks whether the token list contains the unknow token
    #it appears when the model cannot recognize a word in the sentence
    #false means the all the words were recognized  by the tokenizer
    print("contains [UNK]? :", (tokenizer.unk_token_id in ids))
    

#creates a list of lnaguage names from the keys of the dictionary
langs = list(sent_simple.keys())
#creates a list of sentence in the same order as the languages
sents = [sent_simple[l] for l in langs]
#calls the batch_embeddings function
#converts each sentence into its embedding vector
embs = batch_embeddings(sents)

#computes the cosine similarity between every pair of embeddings
#since we use the same matrix twice we get a square matrix of shape(3,3)
sim_mat = pairwise_sim(embs,embs)
#prints a label for readability
print("Pairwise Similarity Matrix ( rows/cols = Hindi,Bhojpuri,Magahi):")
#prints the similarity matrix rounded to 3 decimal places
print(np.round(sim_mat,3))

#defines the list of the specific language pairs to be compared
pairs = [
    ("Hindi", "Bhojpuri"),
    ("Hindi", "Magahi"),
    ("Bhojpuri", "Magahi")   
]

#creates a dictionary mapping, used to access the correct row/column in the similarity matrix
indices = {lang:i for i,lang in enumerate(langs)}
#loop through each language pair
#find their indices in the similarity matrix
#extrac the similarity value at position i,j
#prints the value
for a,b in pairs:
    i,j = indices[a],indices[b]
    print(f"{a}<->{b}:{sim_mat[i,j]:3f}")
    
#a python list is created for each language
#each list contains 3 simple sentences in that language
#they are semantically similar but linguistically different
#used to compute embeddings in batch and compare similarities
h_sentences = ["बिजली चली गई।", "जैसी करनी वैसी भरनी।","तुम कैसे हो?"]
b_sentences =  ["बिजली बंद हो गईल।", "बस अइसहीं।", "तू कइसे हव?"]
m_sentences = ["बिजली गुल हो गेल।","अइसहीं।","कइसन हें?"]

#concatenates the 3 lists
#becomes a single list of 9 sentence in this order
#1-3 Hindi, 4-6 Bhojpuri, 7-9 Magahi
all_sents = h_sentences + b_sentences + m_sentences
#used to know which sentence belongs to which language
labels = (["Hindi"]*len(h_sentences)+["Bhojpuri"]*len(b_sentences)+["Magahi"]*len(m_sentences))

#converts each sentence to its cls embedding
#output a matrix of shape(9,embedding_dimension) here(9,768)
all_embs = batch_embeddings(all_sents)

#computes the average cosine similarity between two language groups
#groups: list of pairs eg(Hindi,Bhojpuri)
#embeddings: 2-D matrix of all embeddings
#labels: the labe for each embedding
def mean_pairwise_between(groups,embeddings,labels):
    #an empty dictionary is created to store the results
    out ={}
    #loops through each language pair
    #like ("Hindi","Bhojpuri")
    for A,B in groups:
        #creates a list of indices for all embeddings belonging to a particular language A
        #for eg A="Hindi" this gives [0,1,2]
        idxA = [i for i,l in enumerate(labels)if l==A]
        #same as above but for language B
        idxB = [i for i,l in enumerate(labels)if l==B]
        #extracts embeddings of groupp A and B
        #computes a pairwise similarity matrix between them
        #if each group has 3 sentences then matrix is (3,3)
        sim = pairwise_sim(embeddings[idxA],embeddings[idxB])
        #computes the average cosine similarity between all pairs
        #stores the result in out dictionart with key as (A,B)
        out[(A,B)] = sim.mean()
    #returns a dictionary mapping
    return out

#define the 3 language pairs to be compared
groups = [("Hindi","Bhojpuri"),("Hindi", "Magahi"),("Bhojpuri","Magahi")]
#runs the function produces mean similarity for each pair
means = mean_pairwise_between(groups,all_embs,labels)
#prints a title
print("\n Mean Similarities(stage 2):")

3#loops through the dictionary and prints each pair and its mean similarity rounded to 3 decimals
for k,v in means.items():
    print(k,round(v,3))
    
#a function that finds the most similar and least similar sentences between two languages
#labels,embedding, lnaguage A and language B
#k=number of top pairs to return
def top_k_pairs_between(labels,embeddings,A,B,k=5):
    #indices of embeddings for language A
    idxA = [i for i, l in enumerate(labels)if l==A]
    #indices of embeddings for language B
    idxB = [i for i, l in enumerate(labels)if l==B]
    #computes similarity matrix between embeddings of language A and B
    sims = pairwise_sim(embeddings[idxA],embeddings[idxB])
    #empty list stors flattened similarity info
    flat = []
    #loop through every pair of indices
    #embedding from A, embedding from B
    for ii,i in enumerate(idxA):
        for jj,j in enumerate(idxB):
            #for each pair store a tuple of(similarity, index in A, index in B)
            #to know which two sentences produces the similarity score
            flat.append((sims[ii,jj],i,j))
    #sorts all pairs by similarity value highest first
    flat_sorted= sorted(flat,key=lambda x:x[0],reverse = True)
    #returns two lists
    #first is top k most similar pairs
    #second is bottom k least similar pairs
    return flat_sorted[:k],flat_sorted[-k:]

#calls the function top_k_pairs_between for Hindi and Bhojpuri
#labels:list of language labels for each sentence
#all_embs: Numpy array of embeddings shape
#A=Hindi, B=Bhojpuri
#k=3 request the top 3 most similar and bottom 3 least similar pairs
top,bottom = top_k_pairs_between(labels,all_embs,"Hindi","Bhojpuri",k=3)
#print the top lists
print("\nTop Hindi-Bhojpuri pairs(sim,idx_hindi,idx_bho):",top)
#print the bottom list
print("Bottom Hindi-Bhojpuri pairs:",bottom)

#function the visualizes the embeddings in 2D
#embeddings: Numpy array shape(N,hidden)
#labels= list of length N with language labels(eg Hindi,Bhojpuri,Magahi)
#method:string, pca or tsne
#perplexity:parameter for tsne
def plot_2d(embeddings,labels,method="pca",perplexity=30):
    #creates a PCA object with 2 components
    #fits the PCA to embeddings and transforms them to 2D
    #proj becomes a numpy array of shape(N,2) where each row is the (x,y) coordinate of a sentence in 2D space
    if method == "pca":
        proj = PCA(n_components=2).fit_transform(embeddings)
    
    #creates a TSNE object with 2 components
    #perplexity=perplexity controls effective number of neighbours
    #init initializes t-SNE with PCA projection for stability
    #fit_transforms computes the 2D projection
    else:
        proj = TSNE(n_components=2,perplexity=perplexity,init="pca").fit_transform(embeddings)
    
    #A small dictionary mapping label strings to matplotlib color codes
    #determines the color to be used to plot for each language points
    colors = {"Hindi":"C0","Bhojpuri":"C1","Magahi":"C2"}
    #a new matplotlib figure is created with size 8 by 6 inches
    plt.figure(figsize=(8,6))
    #set(labels)returns the unique labels
    #sorted makes the order deterministic(here alphabetical)
    #loop var lab iterates over each unique label
    for lab in sorted(set(labels)):
        #for the current label lab build a list of indices
        #these indices select the rows of pro/embeddings that belong to the current language
        idx=[i for i,l in enumerate(labels)if l==lab]
        #plot the points using scatter plot
        #proj[idx,0]= x coordinate for selected indices
        #proj[idx,1]= y coordinate for selected indices
        #c=colors[lab] colors for this language
        #alpha is the point transparency, 1 means opaque 0 means fully transparent
        #label=lab label used by the legend
        plt.scatter(proj[idx,0],proj[idx,1],c=colors[lab],alpha=0.8,label=lab)
        
    #adds a legend to the plot showing which color corresponds to which language
    plt.legend()
    #set the plot title
    plt.title(f"2D projection({method}) of embeddings")
    #displays the figure
    plt.show()
    
#calls the function to plot the 2D projection using PCA
plot_2d(all_embs,labels,method="pca")


