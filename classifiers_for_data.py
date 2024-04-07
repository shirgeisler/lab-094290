import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# ignore warnings
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")


def before_pca():
    # load the data
    data = pd.read_csv('people_embedded_and_scored.csv', header=0)
    data_with_full_info = data.copy()
    # remove all of the rows in which the above value is 'about'
    data = data[data['about'] != 'about']
    # whenever X_explanation is in the form of 'no ... provided', change X_score to None
    # get all the columns that end with _exaplanation
    explanation_columns = [col for col in data.columns if col.endswith('_explanation')]
    for col in explanation_columns:
        name = col.split('_')[0]
        data.loc[data[col].str.contains('no .* provided', case=False), name + '_score'] = None

    # inverse the non-inverted columns
    cols_to_invert = ['About Section', 'Certifications and Field Alignment', 'Followers and Interaction', 'Position',
                      'Experience Depth and Recommendation Specificity', 'Recommendations',
                      'Content Quality and Engagement', 'Self Promotion', 'Degree']

    # change to numeric first
    for col in cols_to_invert:
        data[col + '_score'] = 6 - pd.to_numeric(data[col + '_score'], errors='coerce')

    # calculate an ultimate score, which is the average of the scores. convert the scores to numeric first
    score_columns = [col for col in data.columns if col.endswith('_score')]
    data[score_columns] = data[score_columns].apply(pd.to_numeric, errors='coerce')
    # ultimate score is mean of all the scores
    data['ultimate_score'] = data[score_columns].mean(axis=1)
    # round the ultimate score to the nearest integer
    data['ultimate_score_categorial'] = data['ultimate_score'].round()
    data.to_csv('people_embedded_and_scored - remove_stuffupdated.csv', index=False)
    # remove all columns that end with _score or _explanation except for ultimate_score)

    # remove columns about	certifications	city	comments	degree	duration_short_numeric	field	followers	id	position	post_content	post_title	posts	recommendations url
    columns_to_remove = ['about', 'certifications', 'city', 'comments', 'degree', 'duration_short_numeric', 'field',
                         'followers', 'position', 'post_content', 'post_title', 'posts', 'recommendations', 'url']
    data = data.drop(columns=columns_to_remove)
    # in all of the columns that end with "_embedding", turn zero lists with length 384 into None
    embedding_columns = [col for col in data.columns if col.endswith('_embedding')]

    def parse_string_to_array(s):
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]  # Remove the square brackets
            # remove commas
            s = s.replace(',', '')
            return np.array([float(x) for x in s.split()])
        else:
            return None

    for col in embedding_columns:
        # Filter out rows where all elements in the vector/list are zero
        data[col] = data[col].apply(parse_string_to_array)
        # create columns for each of the elements in the embedding
        for i in range(384):
            data[f'{col}_{i}'] = data[col].apply(lambda x: x[i] if float(x[i]) != 0 else None)
    data.to_csv('people_embedded_and_scored - to_algorithms.csv', index=False)

    return data_with_full_info


def pca():
    data = pd.read_csv('people_embedded_and_scored - to_algorithms.csv', header=0)
    columns_to_remove = ['about_embedding', 'position_embedding', 'certifications_embedding', 'post_content_embedding',
                         'post_title_embedding', 'field_embedding', 'recommendations_embedding', 'degree_embedding',
                         'comments_embedding']
    data = data.drop(columns=columns_to_remove)

    # for every column that ends with _count, fill the missing values with 0
    count_columns = [col for col in data.columns if col.endswith('_count')]
    data[count_columns] = data[count_columns].fillna(0)

    # fill missing values with the mean for all the columns that dont end with _score
    non_score_columns = [col for col in data.columns if not col.endswith('_score')]
    data[non_score_columns] = data[non_score_columns].fillna(data[non_score_columns].mean())

    embedding_columns = [col for col in data.columns if 'embedding' in col]
    # Ensure data is in the correct format for PCA
    pca_input = data[embedding_columns].to_numpy(dtype='float64')

    # Perform PCA
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(pca_input)

    # Create a new DataFrame for PCA results and non-embedding data
    pca_columns = [f'PCA_{i + 1}' for i in range(pca.n_components)]
    pca_df = pd.DataFrame(pca_result, columns=pca_columns)
    # reattach the non-embedding data
    non_embedding_columns = [col for col in data.columns if 'embedding' not in col]
    result_df = pd.concat([data[non_embedding_columns], pca_df], axis=1)

    # Optionally, save the result to a new CSV file
    result_df.to_csv('people_embedded_and_scored_after_PCA.csv', index=False)


def scale_features(X_train, X_test):
    # scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def get_weight_for_class(y_train):
    # get the frequency of each class
    class_freq = y_train.value_counts(normalize=True)
    # get the weight for each class
    weights = {score: 1 / class_freq[score] for score in class_freq.index}
    return weights


if __name__ == '__main__':
    data_full_info = before_pca()
    pca()
    data_full_info = pd.read_csv('people_embedded_and_scored.csv', header=0)


    to_save = data_full_info[data_full_info['about'] != 'about']
    to_save.to_csv('people_embed_score_clean.csv', index=False)

    data = pd.read_csv('people_embedded_and_scored_after_PCA.csv', header=0)
    measure_list = []
    # for measure in About Section_score	Certifications and Field Alignment_score	Followers and Interaction_score	Position_score	Experience Depth and Recommendation Specificity_score	Recommendations_score	Degree_score	Content Quality and Engagement_score	Self Promotion_score	Attention Seeking_score	Self Glorification_score

    for measure in ['About Section_score', 'Certifications and Field Alignment_score',
                    'Followers and Interaction_score', 'Position_score',
                    'Experience Depth and Recommendation Specificity_score', 'Recommendations_score', 'Degree_score',
                    'Content Quality and Engagement_score', 'Self Promotion_score', 'Attention Seeking_score',
                    'Self Glorification_score', 'ultimate_score_categorial']:

        # remove observations with missing values in the measure column
        data = data.dropna(subset=[measure])

        print('records for model (train+test):', len(data))


        # create a random forest classifier, features are all of the columns except ultimate_score	ultimate_score_categorial
        train, test = train_test_split(data, test_size=0.3, random_state=42)
        print(f'train size: {len(train)}, test size: {len(test)}')
        # features are all the columns that dont have the words score and explanation in them
        features = [col for col in data.columns if 'score' not in col and 'explanation' not in col and col != 'id']
        # remove the columns starting with PCA
        features = [col for col in features if not col.startswith('PCA')]

        X_train = train[features]
        y_train = train[measure].astype(int)
        X_test = test[features]
        y_test = test[measure].astype(int)

        # scale the features
        X_train, X_test = scale_features(X_train, X_test)

        # Combine oversampling and undersampling in a pipeline
        pipeline = Pipeline([
            ('oversample', SMOTE(random_state=42, k_neighbors=2)),
            #('undersample', RandomUnderSampler(random_state=42))
        ])


        #create hist of the scores
        sns.histplot(data[measure])
        plt.title(f'Histogram of {measure}')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        #remove decimals from axes
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.savefig(f'hist_{measure}.png')
        plt.show()


        # Apply the pipeline to the dataset
        try:
            X_train, y_train = pipeline.fit_resample(X_train, y_train)
        except ValueError:
            print(f'Could not resample in measure {measure}')


        #print train size per category
        print('train size per category:', y_train.value_counts())

        weights = get_weight_for_class(y_train)

        # RANDOM FOREST REGRESSOR
        clf = RandomForestRegressor(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train, sample_weight=y_train.map(weights))
        # clf.fit(X_train, y_train)

        print('Random Forest Regressor - TEST SET ' + measure)
        # predict on test
        y_pred = clf.predict(X_test)
        # add it to the test dataframe
        test['predicted_score_RF_TEST'] = y_pred
        # get ids of some random person with the highest and lowest scores
        highest_score = test['predicted_score_RF_TEST'].idxmax()
        lowest_score = test['predicted_score_RF_TEST'].idxmin()
        # calculate rmse
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        # calculate accuracy after rounding
        y_pred_rounded = np.round(y_pred)
        accuracy = accuracy_score(y_test, y_pred_rounded)
        print(f'RMSE: {rmse}')
        print(f'Accuracy: {accuracy}')
        print(y_pred)
        print(y_test)
        cm = confusion_matrix(y_test, np.round(y_pred))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.title('Test Set - Random Forest Regressor \n' + measure)
        # set the x and y labels
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'RF_{measure}_test.png')
        plt.show()

        # predict on train
        print('Random Forest Regressor - TRAIN SET ' + measure)
        y_pred = clf.predict(X_train)
        # calculate rmse
        rmse = np.sqrt(np.mean((y_pred - y_train) ** 2))
        # calculate accuracy after rounding
        y_pred_rounded = np.round(y_pred)
        accuracy = accuracy_score(y_train, y_pred_rounded)
        print(f'RMSE: {rmse}')
        print(f'Accuracy: {accuracy}')
        cm = confusion_matrix(y_train, np.round(y_pred))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.title('Train Set - Random Forest Regressor \n' + measure)
        # set the x and y labels
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'RF_{measure}_train.png')
        plt.show()

        # NOW DO LOGISTIC REGRESSION MULTICLASS
        print('Logistic Regression - TEST SET ' + measure)
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
        # clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test['predicted_score_LR_TEST'] = y_pred
        # calculate accuracy after rounding
        accuracy = accuracy_score(y_test, y_pred)
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        print(f'Accuracy: {accuracy}')
        print(f'RMSE: {rmse}')
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.title('Test Set - Logistic Regression \n' + measure)
        # set the x and y labels
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'LR_{measure}_test.png')
        plt.show()

        print('Logistic Regression - TRAIN SET ' + measure)
        # clf = LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced')
        y_pred = clf.predict(X_train)
        # train['predicted_score_LR_TRAIN'] = y_pred
        # calculate accuracy after rounding
        accuracy = accuracy_score(y_train, y_pred)
        print(f'Accuracy: {accuracy}')
        cm = confusion_matrix(y_train, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.title('Train Set - Logistic Regression \n' + measure)
        # set the x and y labels
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'LR_{measure}_train.png')
        plt.show()

        # DO LINEAR REGRESSION
        print('Linear Regression')
        from sklearn.linear_model import LinearRegression

        clf = LinearRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test['predicted_score_LINR_TEST'] = y_pred
        # calculate accuracy after rounding
        # round and truncate to 0 and 5
        y_pred = np.clip(np.round(y_pred), 1, 5)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        cm = confusion_matrix(y_test, np.round(y_pred))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.title('Test Set - Linear Regression \n' + measure)
        # set the x and y labels
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'LINR_{measure}_test.png')
        plt.show()

        print('Linear Regression - TRAIN SET ' + measure)
        y_pred = clf.predict(X_train)
        # train['predicted_score_LINR_TRAIN'] = y_pred
        # calculate accuracy after rounding
        y_pred = np.clip(np.round(y_pred), 1, 5)
        accuracy = accuracy_score(y_train, y_pred)
        print(f'Accuracy: {accuracy}')
        cm = confusion_matrix(y_train, np.round(y_pred))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.title('Train Set - Linear Regression \n' + measure)
        # set the x and y labels
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'LINR_{measure}_train.png')
        plt.show()

        # do SVM
        print('SVM - TEST SET \n' + measure)
        from sklearn.svm import SVC

        clf = SVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # test['predicted_score_SVM_TEST'] = y_pred
        # calculate accuracy after rounding
        accuracy = accuracy_score(y_test, np.round(y_pred))
        print(f'Accuracy: {accuracy}')
        RMSE = np.sqrt(np.mean((y_pred - y_test) ** 2))
        print(f'RMSE: {RMSE}')

        cm = confusion_matrix(y_test, np.round(y_pred))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.title('Test Set - SVM \n' + measure)
        # set the x and y labels
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'SVM_{measure}_test.png')
        plt.show()


        print('SVM - TRAIN SET \n' + measure)
        y_pred = clf.predict(X_train)
        # train['predicted_score_SVM_TRAIN'] = y_pred
        # calculate accuracy after rounding
        accuracy = accuracy_score(y_train, np.round(y_pred))
        print(f'Accuracy: {accuracy}')

        cm = confusion_matrix(y_train, np.round(y_pred))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.title('Train Set - SVM \n' + measure)
        # set the x and y labels
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'SVM_{measure}_train.png')
        plt.show()

    # NOW DO KMEANS ON THE ENTIRE DATASET
    print('KMeans')
    kmeans = KMeans(n_clusters=3)
    # get the features
    X = data[features]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # do pca to dimension 5 now
    pca = PCA(n_components=5)
    X = pca.fit_transform(X)
    kmeans.fit(X)
    import numpy as np

    # Assuming kmeans is already fitted
    unique_labels = np.unique(kmeans.labels_)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans.labels_, s=1, cmap='viridis')
    centers = kmeans.cluster_centers_
    centers = pca.transform(centers)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    # Generate custom legend labels with cluster numbers
    legend_labels = [f'Cluster {label}' for label in unique_labels]

    plt.xlabel('axis 1')
    plt.ylabel('axis 2')
    plt.title('2D PCA of KMeans Clusters')
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    plt.savefig('kmeans.png')
    plt.show()

    silhouette = silhouette_score(X, kmeans.labels_)
    print(f'Silhouette Score: {silhouette}')
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    ids = data['id']
    people_and_clusters = {}
    for i in range(3):
        cluster_indices = np.where(labels == i)[0]
        distances = np.linalg.norm(X[cluster_indices] - centers[i], axis=1)
        n_closest_indices = np.argsort(distances)[:5]
        ids_to_display = ids.iloc[cluster_indices[n_closest_indices]]
        ids_as_list = ids_to_display.values.tolist()
        print(f'Cluster {i}')
        # print id per row
        for id in ids_as_list:
            print(id)
            people_and_clusters[id] = i

    # get the full data of the people in the clusters
    relevant_rows = data_full_info[data_full_info['id'].isin(people_and_clusters.keys())]
    # attach the clusters
    relevant_rows['cluster'] = relevant_rows['id'].map(people_and_clusters)
    relevant_rows.to_csv(f'people_in_clusters_2_{measure}.csv', index=False)
    print(relevant_rows)
