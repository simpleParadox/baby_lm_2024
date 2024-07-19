"""
Implement the functions for the different scoring functions.
"""
import torch


def loss_score(model, dataset):
    """
    This function calculates the loss of a model on a given dataset.
    """

    # Do preprocessing here for the samples in the dataset.
    # TODO: Call the appropriate function to preprocess the dataset.

    loss_values = []
    with torch.no_grad():
        for batch in dataset:
            loss_values.append(model(batch).loss)



    
    return loss_values


def number_of_objects_score(images):
    """
    This function calculates the number of objects in the images.
    The number of objects is the score.
    Can be bucketed if needed.
    """

    num_objects_scores = []
    for image in images:
        # TODO: Calculate the number of objects in an image using some method.
        num_objects = None # TODO: Call appropriate function to calculate the number of objects in the image.


        num_objects_scores.append(num_objects)
    

    return num_objects_scores



def dependency_parse_tree_score(text, parser_path = None):
    """
    This function calculates the dependency parse tree score of the text.
    """

    if parser_path is None:
        assert False, "Please provide the path to the dependency parser model."

    dependency_parse_tree_scores = []
    depth = None
    for sentence in text:
        # TODO: depth = len(dependency_parse_tree_result) # Call the appropriate function to get the dependency parse tree of the sentence.

        dependency_parse_tree_scores.append(depth)
    
    return dependency_parse_tree_scores





def cross_modal_score(image_embeddings, text, image_model=None, text_model=None):
    """
    This function calculates the cross modal score between images and captions using cosine similarity.


    Image embeddings could be dino_v2 embeddings. 

    The text_model is a model that can be used to generate embeddings for the text. This can be the BabyBERTa model.
    """

    if image_model is not None:
        # TODO: Assert that image_embeddings are not any predefined model embeddings.
        image_embeddings = image_model(image_embeddings)
        
    # TODO: Get the embeddings for the text using the text_model.
    # Preprocess the text before passing it to the model.
    # TODO: Write preprocessing steps, which would be the same as the loss_score function.
    text_embeddings = text_model(text)


    cross_modal_scores = []
    for image_embedding, caption in zip(image_embeddings, text_embeddings):
        # Calculate the cosine similarity between the image and the caption.
        cross_modal_scores.append(torch.nn.functional.cosine_similarity(image_embedding, text_model(caption)))


    
    return cross_modal_scores