import csv
from collections import Counter, defaultdict

import pandas as pd
import os
import ast

DATA_DIR = 'coco_original'
KG_DATA_DIR = 'coco_kg'
OUTPUT_DIR = 'preprocessed'
OUTPUT_DIR_METADATA = 'preprocessed/metadata'

RELATION_TYPES = ['belong_to_category', 'related_to_concept', 'taught_in_level',
                  'taught_in_language', 'has_target_audience']
def create_ratings():
    ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'evaluate_latest.csv'))
    ratings_df = ratings_df[['learner_id', 'course_id', 'learner_rating', 'learner_timestamp']]
    ratings_df = ratings_df.rename(columns={'learner_id': 'uid', 'course_id': 'pid', 'learner_rating': 'rating', 'learner_timestamp': 'timestamp'})
    ratings_df['rating'] = ratings_df['rating'].apply(lambda x: 1)
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])
    ratings_df['timestamp'] = ratings_df['timestamp'].astype('int64') // 10 ** 9
    ratings_df.drop_duplicates(subset=['uid', 'pid'], inplace=True)
    ratings_df = ratings_df.dropna()
    ratings_df['uid'] = ratings_df['uid'].astype('int')
    ratings_df['pid'] = ratings_df['pid'].astype('int')
    ratings_df = k_core(ratings_df, k=10)
    ratings_df.to_csv(os.path.join(OUTPUT_DIR, 'ratings.txt'), index=False, sep='\t')
    return ratings_df

def k_core(ratings_df, k=5):
    print(f"Number of ratings before {k}-core: {ratings_df.shape[0]}")
    print(f"Number of users before {k}-core: {ratings_df['uid'].nunique()}")
    print(f"Number of items before {k}-core: {ratings_df['pid'].nunique()}")

    # K-core decomposition process
    while True:
        # Count how many times each user and item appears
        user_counts = ratings_df['uid'].value_counts()
        item_counts = ratings_df['pid'].value_counts()

        # Filter users and items that appear at least k times
        users_to_keep = user_counts[user_counts >= k].index
        items_to_keep = item_counts[item_counts >= k].index

        # Filter the DataFrame
        new_df = ratings_df[
            ratings_df['uid'].isin(users_to_keep) & ratings_df['pid'].isin(items_to_keep)]

        # Check if the filtering changed the DataFrame size
        if new_df.shape[0] == ratings_df.shape[0]:
            break  # Exit the loop if no changes were made
        ratings_df = new_df

    # Resulting DataFrame after k-core decomposition
    print(f"Number of ratings after: {ratings_df.shape[0]}")
    print(f"Number of users after: {ratings_df['uid'].nunique()}")
    print(f"Number of items after: {ratings_df['pid'].nunique()}")
    return ratings_df

def create_users(ratings_df):
    users_df = ratings_df[['uid']].drop_duplicates()
    users_df['gender'] = -1
    users_df['age'] = -1
    users_df.drop_duplicates(inplace=True)
    users_df.to_csv(os.path.join(OUTPUT_DIR, 'users.txt'), index=False, sep='\t')
    return users_df

def create_products(ratings_df):
    products_df = ratings_df[['pid']].drop_duplicates()
    products_df_metadata = pd.read_csv(os.path.join(DATA_DIR, 'course_latest.csv'))
    products_df_metadata = products_df_metadata[['course_id', 'short_url']]
    products_df_metadata['short_url'] = products_df_metadata['short_url'].apply(lambda x: x[1:-1])
    products_df = pd.merge(products_df, products_df_metadata, how='left', left_on='pid', right_on='course_id')
    products_df.rename(columns={'short_url': 'name'}, inplace=True)
    products_df.drop(columns='course_id', inplace=True)

    # join on products_df pid and product_provider_df course_id to add provider to products_df
    product_provider_df = pd.read_csv(os.path.join(DATA_DIR, 'teach_latest.csv'))
    product_provider_df = product_provider_df[['course_id', 'instructor_id']]
    product_provider_df = product_provider_df.drop_duplicates(subset=['course_id'], keep='first')
    products_df = pd.merge(products_df, product_provider_df, how='left', left_on='pid', right_on='course_id')
    products_df.drop(columns='course_id', inplace=True)
    products_df.rename(columns={'instructor_id': 'provider_id'}, inplace=True)
    products_df['provider_id'] = products_df['provider_id'].fillna(-1).astype(int)
    products_df.drop_duplicates(inplace=True)

    product_to_category_df = pd.read_csv(os.path.join(KG_DATA_DIR, 'kg_category.csv'))
    product_to_category_df = product_to_category_df[product_to_category_df['head'].isin(products_df['pid'])]
    product_to_category_df = product_to_category_df[product_to_category_df['relation'].isin(['firstlevel_categorized_as'])]
    product_to_category_df = product_to_category_df.drop_duplicates()
    products_df = pd.merge(products_df, product_to_category_df, how='left', left_on='pid', right_on='head')
    products_df.drop(columns=['head', 'relation'], inplace=True)
    products_df.rename(columns={'tail': 'genre'}, inplace=True)
    products_df['genre'] = products_df['genre'].fillna('unknown')

    products_df.to_csv(os.path.join(OUTPUT_DIR, 'products.txt'), index=False, sep='\t')
    return products_df

def keep_annotations_above_th(annotations, th=0.25):
    valid_annotations = set()
    for annotation in annotations:
        name, score, url, _, _ = annotation
        if score > th:
            valid_annotations.add((name, url))
    return valid_annotations

def create_kg():
    # Create entities
    products_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'products.txt'), sep='\t')
    # initialise entities_df with products_df ids
    entities_df = pd.DataFrame(products_df['pid'].unique(), columns=['name'])

    # kg_category.csv
    kg_category_df = pd.read_csv(os.path.join(KG_DATA_DIR, 'kg_category.csv'))
    kg_category_df = kg_category_df[kg_category_df['head'].isin(products_df['pid']) & ~kg_category_df['tail'].isin(products_df['pid'])].drop_duplicates()
    categories = kg_category_df['tail'].unique()
    entities_df = pd.concat([entities_df, pd.DataFrame(categories, columns=['name'])], ignore_index=True)


    # course_concept_annotation.csv
    target_audience_df = pd.read_csv(os.path.join(KG_DATA_DIR, 'courses_concept_annotation.csv'))
    target_audience_df = target_audience_df[target_audience_df['course_id'].isin(products_df['pid'])]
    # iterrow on course_concept_annotation_df
    annotations_data = []
    for index, row in target_audience_df.iterrows():
        annotations = ast.literal_eval(row['annotations'])
        valid_annotations = keep_annotations_above_th(annotations)
        for annotation in valid_annotations:
            name, url = annotation
            annotations_data.append([row['course_id'], name, url])
    annotations_df = pd.DataFrame(annotations_data, columns=['pid', 'name', 'url']).drop_duplicates()
    annotation_entities = pd.DataFrame(annotations_df['name'].unique(), columns=['name'])
    entities_df = pd.concat([entities_df, annotation_entities], ignore_index=True)

    # kg_instruction_level.csv
    kg_instruction_level_df = pd.read_csv(os.path.join(KG_DATA_DIR, 'kg_instruction_level.csv'))
    kg_instruction_level_df = kg_instruction_level_df[kg_instruction_level_df['head'].isin(products_df['pid']) & ~kg_instruction_level_df['tail'].isin(products_df['pid'])].drop_duplicates()
    levels_df = kg_instruction_level_df['tail'].unique()
    entities_df = pd.concat([entities_df, pd.DataFrame(levels_df, columns=['name'])], ignore_index=True)


    # kg_instruction_language.csv
    kg_language_df = pd.read_csv(os.path.join(KG_DATA_DIR, 'kg_instruction_language.csv'))
    kg_language_df = kg_language_df[kg_language_df['head'].isin(products_df['pid']) & ~kg_language_df['tail'].isin(products_df['pid'])].drop_duplicates()
    languages_df = kg_language_df['tail'].unique()
    entities_df = pd.concat([entities_df, pd.DataFrame(languages_df, columns=['name'])], ignore_index=True)

    # tar_aud_escolabel.csv
    tar_aud_escolabel_df = pd.read_csv(os.path.join(KG_DATA_DIR, 'tar_aud_escolabel.csv'))
    tar_aud_escolabel_df = tar_aud_escolabel_df[tar_aud_escolabel_df['course_id'].isin(products_df['pid'])]
    annotations_data = []
    for index, row in tar_aud_escolabel_df.iterrows():
        labels = ast.literal_eval(row['label'])
        for label in labels:
            annotations_data.append([row['course_id'], label])
    target_audience_df = pd.DataFrame(annotations_data, columns=['pid', 'name'])
    target_audience_entities_df = pd.DataFrame(target_audience_df['name'].unique(), columns=['name'])
    entities_df = pd.concat([entities_df, target_audience_entities_df], ignore_index=True)
    entities_df['eid'] = range(entities_df.shape[0])
    entities_df['entity'] = entities_df['name']
    entities_df = entities_df[['eid', 'name', 'entity']]
    entities_df.to_csv(os.path.join(OUTPUT_DIR, 'e_map.txt'), index=False, sep='\t')

    # Create relations
    relations_df = pd.DataFrame(RELATION_TYPES, columns=['name'])
    relations_df['kb_relation'] = relations_df['name']
    relations_df['id'] = range(relations_df.shape[0])
    relations_df = relations_df[['id', 'kb_relation', 'name']]
    relations_df.to_csv(os.path.join(OUTPUT_DIR, 'r_map.txt'), index=False, sep='\t')

    # Create kg_final.txt
    entity2eid = dict(zip(entities_df['name'], entities_df['eid']))
    # Map 'head' and 'tail' to entity IDs and set 'relation' for kg_category_df
    kg_category_df['head'] = kg_category_df['head'].map(entity2eid)
    kg_category_df['tail'] = kg_category_df['tail'].map(entity2eid)
    kg_category_df['relation'] = 0

    # Process annotations_df similarly
    annotations_df['pid'] = annotations_df['pid'].map(entity2eid)
    annotations_df['name'] = annotations_df['name'].map(entity2eid)
    annotations_df['relation'] = 1
    # Ensure annotations_df columns align with kg_category_df for concatenation
    annotations_df = annotations_df.rename(columns={'pid': 'head', 'name': 'tail'})
    annotations_df = annotations_df[['head', 'tail', 'relation']]

    # Repeat the process for other DataFrames, ensuring you adjust 'relation' value appropriately
    kg_instruction_level_df['head'] = kg_instruction_level_df['head'].map(entity2eid)
    kg_instruction_level_df['tail'] = kg_instruction_level_df['tail'].map(entity2eid)
    kg_instruction_level_df['relation'] = 2

    kg_language_df['head'] = kg_language_df['head'].map(entity2eid)
    kg_language_df['tail'] = kg_language_df['tail'].map(entity2eid)
    kg_language_df['relation'] = 3

    target_audience_df['pid'] = target_audience_df['pid'].map(entity2eid)
    target_audience_df['name'] = target_audience_df['name'].map(entity2eid)
    target_audience_df['relation'] = 4
    # Adjust tar_aud_escolabel_df columns to align with kg_category_df for concatenation
    target_audience_df = target_audience_df.rename(columns={'pid': 'head', 'name': 'tail'})

    kg_final_df = pd.concat(
        [kg_category_df, annotations_df, kg_instruction_level_df, kg_language_df, target_audience_df],
        ignore_index=True)
    kg_final_df.rename(columns={'head': 'entity_head', 'tail': 'entity_tail'}, inplace=True)
    # At this point, kg_final_df contains all the merged data with entity IDs and relation indices set.
    # Make sure to drop duplicates if necessary
    kg_final_df = kg_final_df.drop_duplicates().reset_index(drop=True)
    kg_final_df.to_csv(os.path.join(OUTPUT_DIR, 'kg_final.txt'), index=False, sep='\t')

    # Create i2kg_map.txt
    pid2eid = dict(zip(entities_df['name'], entities_df['eid']))
    pid2kg_df = pd.DataFrame(products_df['pid'], columns=['pid'])
    pid2kg_df['eid'] = pid2kg_df['pid'].map(pid2eid)
    pid2kg_df['name'] = products_df['name']
    pid2kg_df['entity'] = pid2kg_df['pid']
    pid2kg_df = pid2kg_df[['eid', 'pid', 'name', 'entity']]
    pid2kg_df.to_csv(os.path.join(OUTPUT_DIR, 'i2kg_map.txt'), index=False, sep='\t')

def time_based_train_test_split(dataset_name, train_size, valid_size):
    uid2pids_timestamp_tuple = defaultdict(list)
    with open(os.path.join(OUTPUT_DIR, 'ratings.txt'), 'r') as ratings_file:  # uid	pid	rating	timestamp
        reader = csv.reader(ratings_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            uid, pid, rating, timestamp = row
            uid2pids_timestamp_tuple[uid].append([pid, int(timestamp)])
    ratings_file.close()

    for uid in uid2pids_timestamp_tuple.keys():
        uid2pids_timestamp_tuple[uid].sort(key=lambda x: x[1])

    train, valid, test = {}, {}, {}
    for uid, pid_time_tuples in uid2pids_timestamp_tuple.items():
        n_interactions = len(pid_time_tuples)
        train_end = int(n_interactions * train_size)
        valid_end = train_end + int(n_interactions * valid_size)+1
        train[uid], valid[uid], test[uid] = pid_time_tuples[:train_end], pid_time_tuples[train_end:valid_end], pid_time_tuples[valid_end:]

    for set_filename in [(train, "train.txt"), (valid, "valid.txt"), (test, "test.txt")]:
        set_values, filename = set_filename
        with open(os.path.join(OUTPUT_DIR, filename), 'w') as set_file:
            writer = csv.writer(set_file, delimiter="\t")
            for uid, pid_time_tuples in set_values.items():
                for pid, time in pid_time_tuples:
                    writer.writerow([uid, pid, 1, time])
        set_file.close()

def add_products_metadata():
    products_df = pd.read_csv(os.path.join(OUTPUT_DIR, "products.txt"), sep="\t")

    #Add item popularity
    interactions_df = pd.read_csv(os.path.join(OUTPUT_DIR, "train.txt"), sep="\t", names=["uid", "pid", "interaction", "timestamp"])
    product2interaction_number = Counter(interactions_df.pid)
    most_interacted = max(product2interaction_number.values())
    less_interacted = 0 if len(list(product2interaction_number.keys())) != products_df.pid.unique().shape[0] \
        else min(product2interaction_number.values())
    for pid in list(products_df.pid.unique()):
        occ = product2interaction_number[pid] if pid in product2interaction_number else 0
        product2interaction_number[pid] = (occ - less_interacted) / (most_interacted - less_interacted)

    products_df.insert(3, "pop_item", product2interaction_number.values(), allow_duplicates=True)

    #Add provider popularity
    item2provider = dict(zip(products_df.pid, products_df.provider_id))
    interaction_provider_df = interactions_df.copy()
    interaction_provider_df['provider_id'] = interaction_provider_df.pid.map(item2provider)
    provider2interaction_number = Counter(interaction_provider_df.provider_id)
    provider2interaction_number[-1] = 0
    most_interacted, less_interacted = max(provider2interaction_number.values()), min(provider2interaction_number.values())
    for pid in provider2interaction_number.keys():
        occ = provider2interaction_number[pid]
        provider2interaction_number[pid] = (occ - less_interacted) / (most_interacted - less_interacted)
    products_df["pop_provider"] = products_df.provider_id.map(provider2interaction_number)
    products_df = products_df[["pid", "name", "provider_id", "genre", "pop_item", "pop_provider"]]
    products_df.to_csv(os.path.join(OUTPUT_DIR, "products.txt"), sep="\t", index=False)

if __name__ == '__main__':
    ratings_df = create_ratings()
    users_df = create_users(ratings_df)
    products_df = create_products(ratings_df)
    create_kg()
    time_based_train_test_split('coco', 0.6, 0.2)
    add_products_metadata()