import numpy as np
import pandas as pd
import os
import logging

os.environ['API_KEY'] = 'AIzaSyDtvCGZXE5IyzmepEPvqg_qNZpq54K4Ccc'
import google.generativeai as genai
import json
import ast
import sentence_transformers
import nltk
from nltk.tokenize import word_tokenize
import emoji
from google.api_core import exceptions
import time


def load_data(file_name):
    data = pd.read_csv(file_name)
    # convert the columns that contain lists to actual lists
    data['certifications'] = data['certifications'].apply(json.loads)
    data['post_content'] = data['post_content'].apply(json.loads)
    data['post_title'] = data['post_title'].apply(json.loads)
    data['comments'] = data['comments'].apply(json.loads)
    data['recommendations'] = data['recommendations'].apply(json.loads)
    data['degree'] = data['degree'].apply(json.loads)
    data['field'] = data['field'].apply(json.loads)

    # convert comments dict keys to integers
    data['comments'] = data['comments'].apply(lambda x: {int(k): v for k, v in x.items()})

    return data


def load_data_ast_literal_eval(file_name):
    data = pd.read_csv(file_name)
    # convert the columns that contain lists to actual lists
    data['certifications'] = data['certifications'].apply(ast.literal_eval)
    data['post_content'] = data['post_content'].apply(ast.literal_eval)
    data['post_title'] = data['post_title'].apply(ast.literal_eval)
    data['comments'] = data['comments'].apply(ast.literal_eval)
    data['recommendations'] = data['recommendations'].apply(ast.literal_eval)
    data['degree'] = data['degree'].apply(ast.literal_eval)
    data['field'] = data['field'].apply(ast.literal_eval)

    # convert comments dict keys to integers
    data['comments'] = data['comments'].apply(lambda x: {int(k): v for k, v in x.items()})

    return data


def call_generative_ai_with_retry(model, prompt):
    for _ in range(3):  # Try up to 3 times
        try:
            response = model.generate_content(prompt)
            return response
        except exceptions.InternalServerError:
            print("Internal server error encountered. Retrying...")
            time.sleep(2)  # Wait for 2 seconds before retrying


def scoring_function(row):
    start_time = time.time()
    about = row['about']
    certifications = row['certifications']
    followers = row['followers']
    position = row['position']
    recommendations = row['recommendations']
    degree = row['degree']
    field = row['field']
    duration_short_numeric = row['duration_short_numeric']
    post_title = row['post_title']
    post_content = row['post_content']
    comments = row['comments']

    post_and_comments_dict = {}
    for i in range(len(post_content)):
        try:
            post_and_comments_dict[f"post_title_{i}"] = post_title[i]
        except:
            post_and_comments_dict[f"post_title_{i}"] = ''
        post_and_comments_dict[f"post_content_{i}"] = post_content[i]
        post_and_comments_dict[f"comments_{i}"] = comments[i]

    prompts = {}
    prompts[
        'About Section'] = "Analyze clarity, conciseness, and specific achievements. Generic descriptions receive higher scores (1 for unclear or overly verbose content)."
    prompts[
        'Certifications and Field Alignment'] = "Assess relevance to expertise and professional credibility. Penalize irrelevant certifications (1 for irrelevant certifications)."
    prompts[
        'Followers and Interaction'] = "Evaluate engagement quality in relation to follower count. Penalize low engagement (1 for low engagement, especially with high follower count)."
    prompts[
        'Position'] = "Ensure alignment between job title and content. Penalize lack of alignment (1 for inconsistency)."
    prompts[
        'Experience Depth and Recommendation Specificity'] = "Measure depth based on specificity and quantifiability of recommendations (1 for lack of depth)."
    prompts[
        'Recommendations'] = "Look for specific examples and quantifiable achievements. Penalize generic endorsements (1 for generic recommendations)."
    prompts[
        'Degree'] = "Consider institution prestige and relevance. Penalize lack of prestige or relevance (1 for no relevance or prestige)."
    prompts[
        'Content Quality and Engagement'] = "Assess clarity, relevance, and originality. Penalize unoriginal, irrelevant, non-engaging content (1 for unengaging content)."
    prompts[
        'Self Promotion'] = "Evaluate the level of repetitive, or unjustified self promotion of the user by observing their about section and their posts (1 for overly self-promotional)."
    prompts[
        'Attention Seeking'] = "Evaluate tendency to seek attention in the about section and in posts through unrelated hot topics, excessive use of emojis and superlatives (5 for seeking attention)."
    prompts[
        'Self Glorification'] = "Evaluate the level of unnecessary self-glorification in the about section and in posts by observing incidents in which the user repeatedly praises themselves (1 for self-glorification)."
    scores_and_explanations = {}

    for key, value in prompts.items():
        logging.basicConfig(level=logging.INFO)

        # Configure generativeai
        genai.configure(api_key=os.environ['API_KEY'])

        # Define generation configuration and safety settings
        generation_config = {
            "candidate_count": 1,
            "max_output_tokens": 1000,
            "temperature": 0.0,
            "top_p": 0.7,
        }
        safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

        # Create generative model
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        prompt_baseline = f"""You are a recruiter seeking to rate LinkedIn user profiles. 
        You will be given 9 features from a profile. Your task is to assess the credibility 
        and depth of their professional experience by assigning a discrete numerical score.
        
        The feature you will be evaluating is "{key}". {value}
        
        The output should be an explanation of the score given (less than 10 words), and the score itself (scale of 1 to 5).
        
        Example Output (important for later parsing):
        The content lacks engagement, but is original | 3
        If no relevant information is provided, please state in the explanation.        
        
        Please proceed with evaluating the LinkedIn profile features provided below, using the criteria and format outlined.
    
        About: {about}
        Certifications: {certifications}
        Followers: {followers}
        Position: {position}
        Recommendations: {recommendations}
        Degree: {degree}
        Field: {field}
        Total months of Experience: {duration_short_numeric}
        Posts and Comments: {post_and_comments_dict}
        """

        response = call_generative_ai_with_retry(model, prompt_baseline)

        if response:
            # take the number that comes after the word "score"
            try:
                score = response.text.split('|')[1].split()[-1]
                explanation = response.text.split('|')[0].replace('explanation - ', "")[:-1]
                scores_and_explanations[key] = {'score': score, 'explanation': explanation}
            except Exception as e:
                error_message = 'An error occurred while parsing the response:' + str(e)
                print(str(e))
                # print('key:', key, '| score:', None, '| explanation:', 'No information provided', '| error:', error_message)
                scores_and_explanations[key] = {'score': None, 'explanation': error_message}
        else:
            error_message = 'An error occurred while generating the response'
            scores_and_explanations[key] = {'score': None, 'explanation': error_message}

    list_of_scores = [scores_and_explanations[key]['score'] for key in scores_and_explanations.keys()]

    list_of_explanations = [scores_and_explanations[key]['explanation'] for key in scores_and_explanations.keys()]

    runtime = time.time() - start_time
    # if the runtime is less than 12 seconds, sleep for the remaining time
    # convert runtime to seconds
    runtime = int(runtime)
    print(f'Runtime: {runtime} seconds')
    if runtime < 12:
        time.sleep(12 - runtime)
        print('Slept for', 12 - runtime, 'seconds')

    return list_of_scores + list_of_explanations


def get_scores_with_gemini_prompts(data):
    # apply the scoring function to each row in the data
    list_of_evals = ['About Section', 'Certifications and Field Alignment', 'Followers and Interaction', 'Position',
                     'Experience Depth and Recommendation Specificity', 'Recommendations', 'Degree',
                     'Content Quality and Engagement', 'Self Promotion', 'Attention Seeking', 'Self Glorification']
    list_of_scores = [f'{score}_score' for score in list_of_evals]
    list_of_explanations = [f'{score}_explanation' for score in list_of_evals]
    feats_to_add = list_of_scores + list_of_explanations
    data[feats_to_add] = data.apply(lambda x: pd.Series(scoring_function(x)), axis=1)
    return data


def embed_simple_sentence(sentence, model):
    # if sentence is not nan
    try:
        return model.encode(sentence)
    except:
        return [0] * 384


def embed_list_of_sentences(sentences, model):
    # get the average embedding of the sentences
    if len(sentences) == 0:
        return [0] * 384
    return np.mean([embed_simple_sentence(sentence, model) for sentence in sentences], axis=0)


def embed_dictionary_of_sentences(dictionary, model):
    # get the average embedding of the sentences
    if len(dictionary) == 0:
        return [0] * 384
    return np.mean([embed_list_of_sentences(sentence, model) for sentence in dictionary.values()], axis=0)


def embed_sentence_of_data(data):
    model = sentence_transformers.SentenceTransformer('paraphrase-MiniLM-L6-v2')

    sentence_atts = ['about', 'position']

    for att in sentence_atts:
        data[f'{att}_embedding'] = data[att].apply(lambda x: embed_simple_sentence(x, model))

    list_atts = ['certifications', 'post_content', 'post_title', 'field', 'recommendations', 'degree']
    for att in list_atts:
        # embed the list of sentences
        data[f'{att}_embedding'] = data[att].apply(lambda x: embed_list_of_sentences(x, model))

    data['comments_embedding'] = data['comments'].apply(lambda x: embed_dictionary_of_sentences(x, model))

    return data


def word_att_counter_sentence(sentence):
    superlatives_count = 0
    self_ref_count = 0
    achievement_success_count = 0
    if pd.isnull(sentence):
        return superlatives_count, self_ref_count, achievement_success_count
    tokenized = word_tokenize(sentence)
    # do part of speech tagging
    pos = nltk.pos_tag(tokenized)
    # count superlatives
    superlatives = ['JJR', 'JJS']

    self_reference_words = ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", "me", "my",
                            "mine", "myself",
                            "us", "our", "ours", "ourselves"]

    achievement_success = ["achievement", "accomplish", "successful", "winner", "top", "best", "expert",
                           "leader", "excellent", "superior", "outstanding", "great", "exceptional",
                           "extraordinary", "remarkable", "brilliant", "genius", "master", "talented",
                           "skilled", "gifted", "ambitious", "visionary", "goal-oriented",
                           "driven"]

    for word, tag in pos:
        word = word.lower()
        if tag in superlatives:
            superlatives_count += 1
        if word in self_reference_words:
            self_ref_count += 1
        if word in achievement_success:
            achievement_success_count += 1

    emoji_count = emoji.emoji_count(sentence)

    if emoji_count > 2:
        print(superlatives_count, self_ref_count, achievement_success_count, emoji_count)
    return superlatives_count, self_ref_count, achievement_success_count, emoji_count


def word_att_counter_list(list_of_sentences):
    if len(list_of_sentences) == 0:
        return 0, 0, 0, 0
    return np.mean([word_att_counter_sentence(sentence) for sentence in list_of_sentences], axis=0)


def count_superlatives_and_self_references(data):
    print('single_sentnece')
    data[['about_superlatives_count', 'about_self_ref_count', 'about_achievement_success_count', 'about_emoji_count']] = \
        data['about'].apply(
            lambda x: pd.Series(word_att_counter_sentence(x)))
    print('multi_sentence')
    data[['posts_superlatives_count', 'posts_self_ref_count', 'posts_achievement_success_count', 'posts_emoji_count']] = \
        data['post_content'].apply(
            lambda x: pd.Series(word_att_counter_list(x)))

    return data


def count_list_items(data):
    data['recommendations_count'] = data['recommendations'].apply(lambda x: len(x))
    data['post_count'] = data['post_content'].apply(lambda x: len(x))
    data['certifications_count'] = data['certifications'].apply(lambda x: len(x))
    return data


if __name__ == '__main__':
    data = load_data('people.csv')
    data = embed_sentence_of_data(data)
    data.to_csv('people_embeddings.csv', index=False)
    data = pd.read_csv('people_embeddings.csv')
    data = load_data_ast_literal_eval('people_embeddings.csv')
    data = count_superlatives_and_self_references(data)
    data.to_csv('people_counters.csv', index=False)
    data = load_data_ast_literal_eval('people_counters.csv')
    data = count_list_items(data)


    # sort data by id
    data = data.sort_values(by='id')
    # for each batch of 30 rows, get the scores and append to csv
    batch_size = 5
    starting_index = 0
    for i in range(starting_index, len(data), batch_size):
        max_index = min(i + batch_size, len(data))
        data_batch = data.iloc[i:max_index]
        data_batch = get_scores_with_gemini_prompts(data_batch)
        data_batch.to_csv('people_embedded_and_scored.csv', mode='a', header=True, index=False)
        print(f'Batch done. {max_index}/{len(data)} rows processed.')
