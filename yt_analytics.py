import streamlit as st
from streamlit_option_menu import option_menu

import time
import os
import re
import requests
import smtplib
from email.mime.text import MIMEText

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from warnings import simplefilter
from googleapiclient.discovery import build



def get_csv_from_loc(loc = r"https://raw.githubusercontent.com/zahemen9900/YouTube-Analytics-App/main/YouTube%20Data%20EDA/yt_cluster_data.csv"):
    if 'yt_data' not in st.session_state:
        st.session_state['yt_data'] = pd.read_csv(loc)
    df = st.session_state['yt_data']

    return df



# Model Evaluation functions


# Note: I used `print()` for exception handling instead of `st.write()` or `st.error()` since the user does not need to see the logs of those errors
    
country_abbreviations = {
    'Unknown': 'Unknown', 'US': 'United States', 'IN': 'India', 'BR': 'Brazil', 'MX': 'Mexico', 'RU': 'Russia',
    'PK': 'Pakistan', 'PH': 'Philippines', 'ID': 'Indonesia', 'TH': 'Thailand', 'FR': 'France', 'CO': 'Colombia',
    'IQ': 'Iraq', 'JP': 'Japan', 'EC': 'Ecuador', 'AR': 'Argentina', 'TR': 'Turkey', 'SA': 'Saudi Arabia',
    'SV': 'El Salvador', 'BD': 'Bangladesh', 'GB': 'United Kingdom', 'DZ': 'Algeria', 'ES': 'Spain', 'PE': 'Peru',
    'EG': 'Egypt', 'JO': 'Jordan', 'MA': 'Morocco', 'SG': 'Singapore', 'SO': 'Somalia', 'CN': 'China', 'CA': 'Canada',
    'AU': 'Australia', 'KR': 'South Korea', 'DE': 'Germany', 'NG': 'Nigeria', 'ZA': 'South Africa', 'IT': 'Italy',
    'VN': 'Vietnam', 'NL': 'Netherlands', 'CL': 'Chile', 'MY': 'Malaysia', 'GR': 'Greece', 'SE': 'Sweden',
    'CH': 'Switzerland', 'AT': 'Austria', 'NO': 'Norway', 'DK': 'Denmark', 'NZ': 'New Zealand', 'IE': 'Ireland',
    'PT': 'Portugal', 'CZ': 'Czech Republic', 'HU': 'Hungary', 'PL': 'Poland', 'RO': 'Romania', 'UA': 'Ukraine',
    'BE': 'Belgium', 'AZ': 'Azerbaijan', 'KZ': 'Kazakhstan', 'UZ': 'Uzbekistan', 'IL': 'Israel', 'IS': 'Iceland',
    'FI': 'Finland', 'FJ': 'Fiji', 'PG': 'Papua New Guinea', 'SB': 'Solomon Islands', 'VU': 'Vanuatu', 'TO': 'Tonga',
    'WS': 'Samoa', 'TV': 'Tuvalu', 'KI': 'Kiribati', 'MH': 'Marshall Islands', 'PW': 'Palau', 'FM': 'Micronesia',
    'NR': 'Nauru', 'TL': 'East Timor',
}
    
continent_mapping = {
    'Unknown': 'Unknown',
    'United States': 'North America', 'India': 'Asia', 'Brazil': 'South America', 'Mexico': 'North America',
    'Russia': 'Europe', 'Pakistan': 'Asia', 'Philippines': 'Asia', 'Indonesia': 'Asia', 'Thailand': 'Asia',
    'France': 'Europe', 'Colombia': 'South America', 'Iraq': 'Asia', 'Japan': 'Asia', 'Ecuador': 'South America',
    'Argentina': 'South America', 'Turkey': 'Asia', 'Saudi Arabia': 'Asia', 'El Salvador': 'North America',
    'Bangladesh': 'Asia', 'United Kingdom': 'Europe', 'Algeria': 'Africa', 'Spain': 'Europe', 'Peru': 'South America',
    'Egypt': 'Africa', 'Jordan': 'Asia', 'Morocco': 'Africa', 'Singapore': 'Asia', 'Somalia': 'Africa',
    'China': 'Asia', 'Canada': 'North America', 'Australia': 'Oceania', 'South Korea': 'Asia', 'Germany': 'Europe',
    'Nigeria': 'Africa', 'South Africa': 'Africa', 'Italy': 'Europe', 'Vietnam': 'Asia', 'Netherlands': 'Europe',
    'Chile': 'South America', 'Malaysia': 'Asia', 'Greece': 'Europe', 'Sweden': 'Europe', 'Switzerland': 'Europe',
    'Austria': 'Europe', 'Norway': 'Europe', 'Denmark': 'Europe', 'New Zealand': 'Oceania', 'Ireland': 'Europe',
    'Portugal': 'Europe', 'Czech Republic': 'Europe', 'Hungary': 'Europe', 'Poland': 'Europe', 'Romania': 'Europe',
    'Ukraine': 'Europe', 'Belgium': 'Europe', 'Azerbaijan': 'Asia', 'Kazakhstan': 'Asia', 'Uzbekistan': 'Asia',
    'Israel': 'Asia', 'Iceland': 'Europe', 'Finland': 'Europe', 'Argentina': 'South America', 'Brazil': 'South America',
    'Colombia': 'South America', 'Mexico': 'North America', 'Peru': 'South America', 'Venezuela': 'South America',
    'Cuba': 'North America', 'Jamaica': 'North America', 'Honduras': 'North America', 'Nicaragua': 'North America',
    'Panama': 'North America', 'Guatemala': 'North America', 'Costa Rica': 'North America', 'Bolivia': 'South America',
    'Paraguay': 'South America', 'Uruguay': 'South America', 'Guyana': 'South America', 'Suriname': 'South America',
    'French Guiana': 'South America', 'Ecuador': 'South America', 'Chile': 'South America', 'Fiji': 'Oceania',
    'Papua New Guinea': 'Oceania', 'Solomon Islands': 'Oceania', 'Vanuatu': 'Oceania', 'Tonga': 'Oceania',
    'Samoa': 'Oceania', 'Tuvalu': 'Oceania', 'Kiribati': 'Oceania', 'Marshall Islands': 'Oceania', 'Palau': 'Oceania',
    'Micronesia': 'Oceania', 'Nauru': 'Oceania', 'East Timor': 'Asia'
}

popular_countries = [
    'United States', 'India', 'Brazil', 'Mexico', 'Russia', 'Pakistan', 'Philippines', 'Indonesia',
    'Thailand', 'France', 'Colombia', 'Iraq', 'Japan', 'Ecuador', 'Argentina', 'Turkey', 'Saudi Arabia',
    'El Salvador', 'Bangladesh', 'United Kingdom', 'Algeria', 'Spain', 'Peru', 'Egypt', 'Jordan', 'Morocco',
    'Singapore', 'Somalia', 'Canada', 'Germany', 'Italy', 'South Korea', 'Australia', 'Netherlands', 'Chile',
    'South Africa', 'Vietnam', 'Malaysia', 'Israel', 'Belgium', 'Sweden', 'Switzerland', 'Austria', 'Greece',
    'Norway', 'Denmark', 'Poland', 'Ireland', 'Portugal', 'Ukraine', 'India', 'Brazil', 'Mexico', 'Russia',
    'Pakistan', 'Philippines', 'Indonesia', 'Thailand', 'France', 'Colombia', 'Iraq', 'Japan', 'Ecuador',
    'Argentina', 'Turkey', 'Saudi Arabia', 'El Salvador', 'Bangladesh', 'United Kingdom', 'Algeria', 'Spain',
    'Peru', 'Egypt', 'Jordan', 'Morocco', 'Singapore', 'Somalia', 'Nigeria', 'Kenya', 'Ghana', 'South Africa',
    'Ethiopia', 'Uganda', 'Tanzania', 'Malawi', 'Zimbabwe', 'Zambia', 'Mozambique', 'Angola', 'Congo', 'Niger',
    'Mali', 'Mauritania', 'Senegal', 'Benin', 'Burkina Faso', 'Sierra Leone', 'Liberia', 'Guinea', 'Togo'
]



#function to retrieve youtube channel info
@st.cache_data
def extract_channel_info(url: str, category: str):
    """
    Extracts information about a YouTube channel, including subscriber count, country, continent, and video statistics.

    Parameters:
    - url (str): The URL of the YouTube channel.
    - category (str): The category or genre of the YouTube channel.

    Returns:
    - pd.DataFrame: A DataFrame containing information about the YouTube channel.

    This function queries the YouTube API to retrieve details about the specified channel, such as the channel's
    username, number of subscribers, country, continent, and average visits and likes for the latest 50 videos.
    The data is presented in a DataFrame for further analysis, and additional information is displayed using
    Streamlit for a user-friendly interface.

    Note: The API key should be stored in a file named 'api_key_lu.txt' in the same directory as this script.

    Example:
    >>> df = extract_channel_info("https://www.youtube.com/channel/UCxyz123", "Technology")
    """

    try:
        api_key = st.secrets['yt_api_key']['api_key']

    
        api_service_name = 'youtube'
        api_version = 'v3'

        url = url.strip('"') #In case of quotation marks
        youtube = build(
            api_service_name, api_version, developerKey = api_key
        )

        channel_id = url.split('/')[-1]  #the part of a channel's URL after the last '/' is the channel_id
        request = youtube.channels().list(
            part = 'snippet, contentDetails, statistics', 
            id = str(channel_id)
        )
        response = request.execute()


        for item in response['items']:
            yt_name = item['snippet']['title']
            yt_thumbnail_url = item['snippet']['thumbnails']['high']['url'] #to get the channel's thumbnail picture
            country_name = country_abbreviations.get(item['snippet']['country'], 'Unknown Country')

            data = {
                'Username': item['snippet']['title'],
                'Subscribers': item['statistics']['subscriberCount'],
                'Categories': category,
                'Country': country_name, 
                'Continent': continent_mapping.get(country_name, 'Unknown Continent')
            }


        # Since the YouTube API for channels can't retrieve video info, we need to make a separate query to get the averga visits and Likes for our channel

        referrer = st.secrets['referrers']['referrer_site'] # Replace this with your own domain name

        # Define the request URL and the headers

        vid_request_url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=50"
        headers = {"Referer": referrer}

        # Make the GET request and print the response
        response_ = requests.get(vid_request_url, headers = headers)
        print('Status code is {}'.format(response_.status_code))

        vid_data = response_.json()

        video_titles, video_ids = [], []   #instantiate empty arrays to collect the video titles and ids

        # Loop through the items list
        for item in vid_data["items"]:
            # Get the video ID and title from the snippet dictionary
            video_id = item["id"]["videoId"]
            video_title = item["snippet"]["title"]

            # Append a tuple of video ID and title to the videos list
            video_titles.append(video_title)
            video_ids.append(video_id)


        request2 = youtube.videos().list(
            part = 'statistics',
            id = ','.join(video_ids) # A comma-separated list of video IDs
        )
        response2 = request2.execute()

        n_visits, n_likes = 0, 0
        # Sum up the like counts for the videos in the current page
        for item in response2['items']:
            n_visits += int(item['statistics']['viewCount'])
            n_likes += int(item['statistics']['likeCount'])

        n_visits /= 50
        n_likes /= 50

        data.update({
            'Visits': n_visits,
            'Likes': n_likes
        })

        yt_channel_df  = pd.DataFrame(data, index = [0])
        yt_channel_df.reindex(['Username', 'Subscribers', 'Category', 'Country', 'Continent', 'Visits', 'Likes']) #make sure the columns are arranged properly
        yt_channel_df = yt_channel_df.astype({
                                  'Username': 'object',
                                  'Subscribers': 'int64',
                                  'Categories': 'object',
                                  'Country': 'object',
                                  'Continent': 'object',
                                  'Visits': 'int64',
                                  'Likes': 'int64'
                              })

        st.write(f"""
                 ##### Hey _**{yt_name}**_, 
                 glad to have you here!
                 """)


        channel_thumbnail = st.image(yt_thumbnail_url, caption = yt_name)

        with st.expander('**Expand to see all your channel info**'):
            st.write('##### _**`channel_info`**_')
            st.write(response)

        st.write("##### Here's a summary of the relevant info:")
        st.write(yt_channel_df)

        with st.expander("_**Expand to see Videos we used**_"):
            formatted_titles = ' '.join([f'<li><b>{title}</b></li>' for title in video_titles])
            st.markdown(
                f"""
                ##### Here's a list of your latest 50 videos:
                ---
                <ul>
                {formatted_titles}
                </ul>
                """, unsafe_allow_html = True)

        return yt_channel_df


    except FileNotFoundError:
        st.write('Error: API key file not found in current directory')
    except Exception as e:
        st.write(f'An error occured: {e}')



        
def score_model(data: pd.core.frame.DataFrame, model, model_params: dict, scaled = False, encoded = False):
    """
    Trains and evaluates a machine learning model using the provided data.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the input features and target variable.
    - model (estimator): The machine learning model to be trained and evaluated.
    - model_params (dict): The hyperparameter grid for model tuning using GridSearchCV.
    - scaled (bool): If False, scales numeric features using StandardScaler (default is False).
    - encoded (bool): If False, encodes categorical features using LabelEncoder (default is False).

    Returns:
    - model_valid (estimator): The trained machine learning model with the best hyperparameters.
    """
    try:
    
        X = data.drop('Username', axis = 1).copy()
        
        y = X.pop('Cluster')
    
        if not scaled:
            scaler = StandardScaler()
            X[X.select_dtypes(include = ['number']).columns] = scaler.fit_transform(X[X.select_dtypes(include = ['number']).columns])
    
        if not encoded:
            for col in X.select_dtypes(include = ['object']).columns:
                label_encoder = LabelEncoder()
                X[col] = label_encoder.fit_transform(X[col])
    
    
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 5)
    
    
        grid_search = GridSearchCV(estimator = model, param_grid = model_params,
                                  scoring = 'accuracy', cv = 5, n_jobs = -1, verbose = 1)
    
        grid_search.fit(X_train, y_train)
        col1, col2 = st.columns([1, 1])

        col1.metric(label = 'Best Cross Validation Score', value = "{:.3f}".format(grid_search.best_score_))

        model_valid = grid_search.best_estimator_
        model_valid.fit(X_train, y_train)
    
        y_preds = model_valid.predict(X_valid)
    
        col2.metric(label = 'Accuracy Score for final model', value = ' {:.3f}'.format(grid_search.score(X_valid, y_valid)))

        st.write("**The best parameters are:**")
        st.write(grid_search.best_params_)
    
        return model_valid

    except Exception as e:
        print(e)


#@st.cache_data()
def make_predictions(df: pd.core.frame.DataFrame, model, yt_channel_df, scaled = False, encoded = False):

    """
    Generates predictions for a YouTube channel using a trained machine learning model.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the input features and target variable used for model training.
    - model (estimator): The trained machine learning model used for making predictions.
    - scaled (bool): If True, scales numeric features using StandardScaler (default is False).
    - encoded (bool): If True, encodes categorical features using LabelEncoder (default is False).
    - **yt_channel_kwargs: Keyword arguments representing the YouTube channel Column Names. Must correspond to the Actual YouTube DataFrame Columns used for Analysis

    Returns:
    - str : A message indicating the predicted cluster for the given YouTube channel.
    """

    try:
        yt_channel_df.rename_axis('Rank', inplace = True)

        if all([col in df.columns for col in yt_channel_df.columns]):
            data = pd.concat([df, yt_channel_df], axis = 0, ignore_index = True)
        else:
            raise ValueError('Error: check that Input arguments match original dataframe columns.')
    
        if not scaled:
            scaler = StandardScaler()
            data[data.select_dtypes(include = ['number']).columns] = scaler.fit_transform(data[data.select_dtypes(include = ['number']).columns])
    
        if not encoded:
            for col in data.select_dtypes(include = ['object']).columns:
                label_encoder = LabelEncoder()
                data[col] = label_encoder.fit_transform(data[col])
    
        prediction = model.predict(data.tail(1).drop(['Username', 'Cluster'], axis = 1))

    
        st.write('Your predicted cluster : **{}**'.format(prediction[-1]))

        return prediction[-1]

    except Exception as e:
        print(e)



#simpler one solely used for recommendations

def score_model_stripped(data: pd.core.frame.DataFrame, model, model_params: dict, scaled = False, encoded = False):
    try:
    
        X = data.drop('Username', axis = 1).copy()
        
        y = X.pop('Cluster')
    
        if not scaled:
            scaler = StandardScaler()
            X[X.select_dtypes(include = ['number']).columns] = scaler.fit_transform(X[X.select_dtypes(include = ['number']).columns])
    
        if not encoded:
            for col in X.select_dtypes(include = ['object']).columns:
                label_encoder = LabelEncoder()
                X[col] = label_encoder.fit_transform(X[col])
    
    
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 5)
    
    
        grid_search = GridSearchCV(estimator = model, param_grid = model_params,
                                  scoring = 'accuracy', cv = 5, n_jobs = -1, verbose = 1)
    
        grid_search.fit(X_train, y_train)

        model_valid = grid_search.best_estimator_
        model_valid.fit(X_train, y_train)
    
        return model_valid

    except Exception as e:
        print(e)



#st.cache_data()
def generate_recommendations(df: pd.core.frame.DataFrame, yt_channel_df, model, scaled=False, encoded=False):

    result = make_predictions(df, model, yt_channel_df, scaled=False, encoded=False)

    cluster_descriptions = {
            1:  """\
                ### 🚀 **Rising Stars Category**

                Congratulations! Your channel belongs to the **Rising Stars** category. This means you're on the fast track to YouTube fame. You've gained millions of subscribers in a short period by creating unique and engaging content that appeals to a large audience. To keep up the momentum and reach the next level, here are some personalized recommendations for you:

                #### Some Popular Names:
                - **Dream:** This Minecraft gamer skyrocketed to fame in 2020 with his innovative and thrilling videos. He is known for his speedruns, manhunts, and collaborations with other popular YouTubers.
                - **Corpse Husband:** This mysterious and deep-voiced narrator started his channel in 2015, but gained massive popularity in 2020 with his horror stories and Among Us gameplay. He is also a singer and songwriter, and has collaborated with celebrities like Machine Gun Kelly.
                - **Emma Chamberlain:** This lifestyle vlogger and influencer rose to prominence in 2018 with her relatable and humorous videos. She has since branched out into podcasting, fashion, and coffee. She was named the "Most Popular YouTuber" by The New York Times in 2019.

                #### Characteristics:
                - Gained millions of subscribers in a short period
                - Created unique and engaging content
                - Appeals to a large audience

                #### Personalized Recommendations:
                - **Content Innovation:** You have a knack for creating **unique and engaging content** that sets you apart from the crowd. Keep experimenting with new ideas and formats, and don't be afraid to try something different. Find out what makes your content special and amplify those elements. For example, you can use analytics to see which videos perform best, and get feedback from your fans to see what they like and want more of.
                - **Social Media Boost:** You can leverage social media platforms to **promote your videos** and grow your fanbase. Engage with your audience on various platforms, such as Instagram, Twitter, TikTok, and Discord. Share behind-the-scenes content, teasers, polls, and giveaways. Interact with your fans by replying to their comments, messages, and tweets. This will help you increase your visibility, loyalty, and reach.
                - **Collaboration Power:** You can collaborate with other creators in your niche to expand your audience and learn from each other. Cross-promotion can lead to rapid growth and exposure. You can also join or create a YouTube group or network with other rising stars, such as the Dream SMP, the Vlog Squad, or the Sister Squad. This will help you build relationships, create more content, and have more fun.
                - **Consistent Effort:** You have the potential to become the next big thing on YouTube, so don't give up on your dreams. Stay consistent and passionate about your content, and your hard work will pay off. 🌟 You can also set goals and track your progress, such as reaching a certain number of views, subscribers, or revenue. Celebrate your achievements and milestones, and reward yourself for your efforts. You can also use tools and resources to help you grow your channel, such as [YouTube Creator Academy](https://www.youtube.com/creators/), [YouTube Analytics](https://studio.youtube.com/?csr=analytics), and [Biteable](https://biteable.com/).
                """
            ,

        2: """\
            ### 📉 **Ground Zero Category**

            Your channel is in the **Ground Zero** category, which means you're probably struggling to get visits, likes, and subscribers. This category is very crowded and competitive, and it's hard to stand out from the rest.

            #### Examples:
            Some Youtuber making it big time in this category are:
            - **The Dodo:** This channel features heartwarming stories of animals and their rescuers. It has over 11 million subscribers and billions of views. 
            - **Tasty:** This channel showcases easy and delicious recipes for all occasions. It has over 21 million subscribers and is one of the most popular food channels on YouTube.
            - **5-Minute Crafts:** This channel offers quick and simple DIY projects, hacks, and tips. It has over 74 million subscribers and is one of the most viewed channels on YouTube.

            #### Characteristics:
            - A lot of videos but low engagement rates
            - Relying on quantity over quality
            - Producing generic or clickbait content

            #### Personalized Recommendations:
            - **Content Strategy:** You need to rethink your **content strategy** and focus on quality over quantity. Instead of uploading a lot of videos that don't get much attention, try to create fewer but better videos that can attract and retain your viewers. Think about what value you can offer to your audience, and what problems you can solve for them. For example, you can use [YouTube Creator Academy](https://www.youtube.com/creators/) to learn how to plan, produce, and optimize your videos.
            - **Audience Targeting:** You need to target a **specific audience** that can relate to your content and engage with it. Instead of trying to appeal to everyone, try to find your niche and your ideal viewer persona. Think about who they are, what they like, what they need, and how you can reach them. For example, you can use [YouTube Analytics](https://studio.youtube.com/?csr=analytics) to understand your audience's demographics, interests, and behavior.
            - **Inspiration Analysis:** You need to analyze successful channels in your niche and get inspiration from them. Instead of copying or competing with them, try to learn from them and see what makes them popular and unique. Think about how you can differentiate yourself and offer something new or better. For example, you can use [Biteable](https://biteable.com/) to compare your channel with other channels and see how you can improve your performance.
            - **Analytics Tools:** You need to use **analytics tools** to measure and improve your channel's performance. Instead of relying on intuition or guesswork, try to use data and insights to guide your decisions and actions. Think about what goals you want to achieve, what metrics you want to track, and what actions you want to take. For example, you can use [Google Analytics](https://studio.youtube.com/?csr=analytics) to monitor and analyze your channel's traffic, conversions, and revenue.
            """
        ,
        3: """\
            ### 🛡️ **Subscribers' Haven Category**

            You belong to the **Subscribers' Haven** category, which means you have a large and loyal fan base that loves your content. However, you also face some challenges in terms of engagement and growth. Here are some tips to help you overcome them and take your channel to the next level.

            #### Examples:
            Some of the most successful YouTube channels in this category are:

            - **PewDiePie**: The king of YouTube, with over 100 million subscribers. He is known for his gaming videos, memes, and commentary.
            - **Mr Beast**: The philanthropist of YouTube, with over hundreds of millions of subscribers. He is known for his extravagant challenges, giveaways, and stunts.

            #### Characteristics:
            The main features of this category are:

            - **Loyal fan bases**: You have a dedicated audience that watches your videos regularly and supports you through various means.
            - **High retention rates**: Your viewers tend to watch your videos for a long time, indicating that they are interested and engaged in your content.
            - **Less frequent posting**: You upload videos less often than other categories, which may affect your visibility and reach.

            #### Recommendations:
            To optimize your performance in this category, you should:

            - **Build a stronger connection with your audience**: You already have a loyal fan base, but you can always make them feel more appreciated and involved. For example, you can interact with them more on social media, respond to their comments, ask for their feedback, or feature them in your videos.
            - **Encourage likes, comments, and shares**: These are the main indicators of engagement on YouTube, and they can help you boost your ranking and exposure. You can ask your viewers to like, comment, and share your videos at the beginning or end of your videos, or use incentives such as giveaways, shoutouts, or polls.
            - **Diversify your content while maintaining uniqueness**: You have a distinctive style and niche that sets you apart from other channels, but you can also explore new topics, genres, or formats that may appeal to your existing or potential viewers. For example, you can collaborate with other creators, try new trends, or experiment with different types of videos such as live streams, podcasts, or shorts.
            - **Keep your supporters entertained and satisfied**: You have the advantage of having a solid foundation of supporters, but you also have to meet their expectations and keep them interested in your content. You can do this by maintaining a consistent quality and frequency of your videos, updating them on your plans and projects, or surprising them with something special or unexpected. 🤝
            """
        ,
        4: """\
                ### ⚖️ **Balancing Act Category**

            You are in the **Balancing Act** category, which means you have a moderate but stable performance on YouTube. You have a decent number of visits, likes, and subscribers, and you create a variety of content that appeals to different audiences. However, you also face some challenges in terms of differentiation and growth. Here are some tips to help you stand out and reach your full potential.

            #### Examples:
            Some of the most popular YouTube channels in this category are:

            - **Katy Perry**: The pop star of YouTube, with over 40 million subscribers. She is known for her music videos, behind-the-scenes, and collaborations with other celebrities.
            - **The Ellen Show**: The talk show of YouTube, with over 38 million subscribers. She is known for her interviews, games, and pranks with famous guests and fans.

            #### Characteristics:
            The main features of this category are:

            - **Decent engagement rates**: You have a fair amount of likes, comments, and shares on your videos, indicating that your viewers are interested and engaged in your content.
            - **Creating a variety of content**: You produce different types of videos, such as entertainment, education, or lifestyle, that cater to different tastes and preferences.
            - **No clear niche or identity**: You do not have a specific focus or theme for your channel, which may make it harder for you to attract and retain a loyal fan base.

            #### Recommendations:
            To optimize your performance in this category, you should:

            - **Maintain a balance in your content**: You have the advantage of being versatile and flexible in your content creation, but you also have to be careful not to lose your identity or direction. You should balance your content between what you are passionate about and what your audience wants to see, and avoid spreading yourself too thin or jumping on every trend.
            - **Explore collaborations with creators in adjacent niches**: You can expand your audience and exposure by collaborating with other creators who have similar or complementary content to yours. For example, you can join forces with other musicians, comedians, or influencers, and create videos that showcase your talents and personalities.
            - **Be consistent in your content delivery**: You can increase your retention and growth rates by uploading videos regularly and consistently. You should establish a schedule and stick to it, and inform your viewers about your plans and updates. You can also use tools such as [YouTube Analytics](https://studio.youtube.com/?csr=analytics) or [Biteable](https://biteable.com/) to track your performance and optimize your content strategy.
            - **Optimize your SEO to improve your visibility and discoverability**: You can boost your ranking and reach on YouTube by using effective keywords, titles, descriptions, tags, and thumbnails for your videos. You should also use catchy and relevant hashtags, and encourage your viewers to subscribe and turn on notifications. You can also learn more about SEO best practices from YouTube Creator Academy or other online resources.
            - **Keep working hard and smart**: You have the potential to reach higher levels of success on YouTube, but you also have to work hard and smart to achieve your goals. You should always strive to improve your content quality and creativity, and keep learning from your feedback and analytics. You should also celebrate your achievements and milestones, and appreciate your supporters. 🚶‍♂️
            """
        ,
        5: """\
            ### 👥 **Engaging Echoes Category**

            You are in the **Engaging Echoes** category, which means you have a high-performance channel that attracts millions of views and likes. You create catchy or trendy content that resonates with a wide audience. However, you also have a low subscriber count compared to other channels, which means you have a challenge in retaining your viewers and building a loyal fan base. Here are some tips to help you turn your viewers into subscribers and grow your community.

            #### Examples:
            Some of the most viral YouTube channels in this category are:

            - **Techno Gamerz**: The gaming sensation of YouTube, with over 20 million subscribers. He is known for his gameplay videos, live streams, and challenges with other gamers.
            - **Kimberly Loaiza**: The queen of YouTube in Latin America, with over 30 million subscribers. She is known for her music videos, vlogs, and collaborations with other influencers.

            #### Characteristics:
            The main features of this category are:

            - **Millions of views and likes**: You have a huge reach and impact on YouTube, with your videos getting millions of views and likes in a short time. You have a knack for creating viral content that appeals to a mass audience.
            - **Catchy or trendy content**: You produce content that is relevant, timely, or entertaining, such as music, comedy, or news. You follow the latest trends and topics, and use effective strategies to capture attention and engagement.
            - **Not as many subscribers as other channels**: You have a lower subscriber count than other channels with similar or lower views and likes. This may indicate that your viewers are not as loyal or committed to your channel, and that they may watch your videos only once or occasionally.

            #### Recommendations:
            To optimize your performance in this category, you should:

            - **Work on strategies to convert viewers into subscribers**: You have the opportunity to grow your subscriber base by converting your viewers into subscribers. You can do this by using clear and compelling calls to action, such as asking your viewers to subscribe and turn on notifications at the beginning or end of your videos, or using pop-ups, cards, or end screens to remind them. You can also use tools such as YouTube Analytics to track your conversion rate and identify areas for improvement.
            - **Consider creating series or themed content to encourage consistent viewership**: You can increase your retention and loyalty rates by creating content that is consistent and coherent, such as series or themed content. For example, you can create a series of videos on a specific topic, genre, or format, and release them on a regular schedule. You can also create themed content based on seasons, events, or occasions, and use catchy titles and thumbnails to generate interest and anticipation. This way, you can keep your viewers hooked and coming back for more.
            - **Engage with your audience through comments and community posts to foster a loyal community**: You can strengthen your relationship with your audience by engaging with them through comments and community posts. You can respond to their comments, ask for their feedback, opinions, or suggestions, or create polls or quizzes to interact with them. You can also use community posts to update them on your plans, projects, or personal life, or share behind-the-scenes, sneak peeks, or teasers of your upcoming videos. This way, you can make your viewers feel valued and involved, and build a loyal community around your channel.
            - **Offer incentives such as giveaways, shoutouts, or merch to reward your fans**: You can motivate and reward your fans by offering them incentives such as giveaways, shoutouts, or merch. You can organize giveaways of products, services, or experiences that are related to your content or niche, and ask your viewers to subscribe, like, comment, or share your videos to enter. You can also give shoutouts to your fans who support you, or feature them in your videos. You can also create and sell merch such as t-shirts, hats, or mugs that represent your brand or personality, and promote them in your videos or social media. This way, you can show your appreciation and gratitude to your fans, and make them feel special and proud. 💬
            """
    }

    return cluster_descriptions.get(result, f"Oops, no specific information available for your cluster 🫤\n Make sure the link is valid, and that your channel's statistics like Country, etc. are properly recorded in your channel's database.")

 

# Create a form for the user to enter their email address
def deliver_recommendations(email_form_key, recommendations):
  st.markdown("""<h4 style = "font-family: Calibri, sans-serif">Get your recommendations delivered straight into your mail! 📩<h4>""", unsafe_allow_html = True)
  with st.form(key= email_form_key):
    email_input = st.text_input('Enter your email address')
    submit_button = st.form_submit_button('Send email')


  if submit_button:
    try:
      # Validate the email input using a regular expression
      email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
      if not re.match(email_pattern, email_input):
        st.error('Invalid email address')
        submit_button = False

      # Create a MIMEText object for the email body
      recommendations = recommendations.replace('#', '').replace('_', '').replace('*', '').replace('[', '').replace(']', '')
      msg = MIMEText(recommendations, "plain")

      app_email = st.secrets['emails']['app_email']
      password = st.secrets['remote_ps']['password']

      # Add the email headers
      msg['From'] = app_email
      msg['To'] = email_input
      msg['Subject'] = 'Your YouTube recommendations'

      # Send the email using Gmail SMTP server
      server = smtplib.SMTP('smtp.gmail.com', 587)
      server.starttls()
      server.login('yt.analytics.app.z@gmail.com', password)
      server.sendmail('yt.analytics.app.z@gmail.com', email_input, msg.as_string())
      server.quit()

      with st.spinner('Sending your recommendations...'):
        time.sleep(1)
      st.success('Email sent successfully!')
      st.write("Please check your spam if you don't see it in youe inbox")

    except Exception as e:
      st.write(e)

#App Config & Other elements

def give_feedback():

  with st.form(key='feedback_form'):
    email_input = st.text_input('Your email address _(so we can reach out to you)_ or just your first name')
    recommendations = st.text_input('Please enter feedback or recommendations here')
    submit_button = st.form_submit_button('Send email')


  while submit_button:
    try:
      if len(recommendations) < 5:
        st.error('Recommendation too short. Please enter a valid response.')
        submit_button = False
      # Create a MIMEText object for the email body

      seconds = time.time()
      localtime = time.localtime(seconds)
      time_sent = time.asctime(localtime)

      recommendations = f"{email_input} gave the following feedback on {time_sent}:\n\n" + recommendations.replace('#', '').replace('_', '').replace('*', '').replace('[', '').replace(']', '')

      msg = MIMEText(recommendations, "plain")

      # Add the email headers
      my_email = st.secrets['emails']['my_email']
      app_email = st.secrets['emails']['app_email']
      password = st.secrets['remote_ps']['password']

      msg['From'] = app_email
      msg['To'] = my_email
      msg['Subject'] = 'Feedback from {}'.format(email_input)

      # Send the email using Gmail SMTP server

      st.write()
      server = smtplib.SMTP('smtp.gmail.com', 587)
      server.starttls()

      server.login(app_email, password)
      server.sendmail(app_email, my_email, msg.as_string())
      server.quit()

      with st.spinner('Sending response...'):
        time.sleep(1)

      st.success('Response received. Thanks for your feedback!😊')
      break

    except Exception as e:
      st.write(e)
      break




# For Navigation Menu and page configurations
    
st.set_page_config(
    page_title = 'YouTube Analytics App',
    page_icon = '🌟',
    initial_sidebar_state = 'collapsed'

    )

selected = option_menu(

    menu_title = None,
    options = ['App Home', 'Summary Stats', 'Top YouTubers in Categories', 'Your Recommendations', 'ML Highlights / About Project'],
    icons = ['cast', 'speedometer', 'stars', 'blockquote-left','info-circle'],
    default_index = 0,
    orientation = 'horizontal'

    )



def main():


    data = get_csv_from_loc()

    with st.sidebar:
        st.markdown(
            """<h1 style = "font-size: 45px; font-family: Arial, sans-serif;">Feedback<h1>"""
            , unsafe_allow_html = True)

        feedback_menu = give_feedback()



    # For Home Page Section;
    if selected == 'App Home':
        col1, col2 = st.columns([.7, .3])

        title = col1.markdown(
            """
            # <div style = "font-size: 80px; font-family: Arial, sans-serif; text-align: left;"><b>YouTube Channel Tip App</b></div>
            """, unsafe_allow_html = True
        )
        yt_icon = col2.markdown(
            """
            <div style = "text-align: top;">
            <img src = "https://cdn-icons-png.flaticon.com/256/1384/1384060.png" width = "300" height = "300" alt = "YouTube Logo"></div>
            """
            , unsafe_allow_html = True
        )
        st.markdown(
            """
            <div style = "padding: 20px;"></div>
            """
            , unsafe_allow_html = True
        )

        description = st.markdown(
            """
            <div style = "font-family: Arial, sans-serif">
            <b><p style = "font-size: 20px;">Do you want to take your YouTube Channel to the next level, but don't know where to start?💭🤔</p><p style = "font-size: 20px;"> You've come to the right place! Here you can get personalized recommendations to boost your likes, subscribers, and visits.⚡📈</p></b>

            <p></p><p></p>
            <p style = "font-size: 20px;">Our app is powered by data from the YouTube API and a machine learning model that analyzes the performance of different types of channels. We will show you some of our insights on the <b style = "color: brown;">YouTube Channel Analytics</b> and how they can help you improve your channel.</p>

            <p style = "font-size: 20px;">Our app will guide you through:</p>
            <ul style = "font-size: 20px;">
            <li>Understanding the data that was used for making recommendations.</li>
            <li>Getting your customized recommendations based on your channel category and goals.</li>
            <li>Understanding the magic behind the predictions and how it works.</li>
            </ul>

            <p style = "font-size: 22px;">Are you ready to grow your channel? Let's get started!✅</p>
            </div>

            <p></p><p></p>
            """,
            unsafe_allow_html=True
        )


    #Model evaluation function
    #I reduced grid after finding optimal parameters, to improve load times

    model_params = {
    'n_estimators': [100],       # Number of trees in the forest
    'max_features': [0.5],     # Number of features to consider at each split
    'min_samples_split': [2],      # Minimum number of samples required to split a node
    'bootstrap': [True]        # Whether to use bootstrap samples when building trees
    }

    if 'rf_model_main' not in st.session_state:
        st.session_state['rf_model_main'] = score_model_stripped(data = data, model = RandomForestClassifier(), model_params = model_params)


    if selected == 'Summary Stats':
        st.write("** _**Hover over plots to reveal info!**_")
        #Country Distribution
        try:
            country_data = data['Country'].value_counts()

            least_10 = country_data.nsmallest(10)

            country_data.drop(least_10.index, inplace = True)

            country_data['Other'] = least_10.sum()

            #country_data = country_data.reset_index().rename(columns = {
            #   'Country': 'Count',
            #   'index': 'Country'
            #   })

            fig = px.pie(country_data.reset_index(), values = 'count', names = 'Country', color_discrete_sequence = px.colors.diverging.Spectral)

            fig.update_layout(
                title_text='<b style = "font-family: Arial, sans-serif">Country Distributions in Data</b>',
                title_font=dict(size=30, family='Arial'),
                title=dict(x=0.5, xanchor='center')
                )
            fig.update_layout(
                autosize = False,
                width = 800,
                height = 700
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.write(f'An error occured: {e}')


        st.write("**How does each Country perform Metric-wise?** 💭")
        #For barplot
        try:
            metric_avgs = data.groupby('Continent')[['Subscribers', 'Visits', 'Likes']].mean().reset_index()

            metric_avgs.rename(columns={
                'Subscribers': 'Average Subscribers per Channel',
                'Visits': 'Average Visits per Channel',
                'Likes': 'Average Likes per Video'
            }, inplace=True)

            fig3 = sp.make_subplots(1, 3)

            fig3.add_trace(go.Bar(x=metric_avgs['Continent'], y=metric_avgs['Average Subscribers per Channel'], marker=dict(color=px.colors.sequential.Inferno)), row=1, col=1)
            fig3.add_trace(go.Bar(x=metric_avgs['Continent'], y=metric_avgs['Average Visits per Channel'], marker=dict(color=px.colors.sequential.Plasma)), row=1, col=2)
            fig3.add_trace(go.Bar(x=metric_avgs['Continent'], y=metric_avgs['Average Likes per Video'], marker=dict(color=px.colors.sequential.Inferno)), row=1, col=3)

            fig3.update_xaxes(title_text='<b style = "font-family: Arial, sans-serif;">Average Subscribers per Channel</b>', row=1, col=1)
            fig3.update_xaxes(title_text='<b style = "font-family: Arial, sans-serif;">Average Visits per Channel</b>', row=1, col=2)
            fig3.update_xaxes(title_text='<b style = "font-family: Arial, sans-serif;">Average Likes per Video</b>', row=1, col=3)

            fig3.update_layout(title_text='<b style = "font-family: Arial, sans-serif;">Metric Performances across Continents</b>',
                        title_font=dict(size=30, family='Arial'),
                        title=dict(x=0.5, xanchor='center'))
            fig3.update_layout(
                autosize=False,
                width = 800,
                height = 650)
            #fig3.update_layout(margin=dict(l=25, r=25, t=100, b=50))

            st.plotly_chart(fig3)

        except Exception as e:
            st.write(f'Error: {e}')



        st.write('**What are the different categories of YouTube Channels?** 🫧')
        # Scatter Matrix
        try:

            dd = data[['Subscribers', 'Likes', 'Visits', 'Cluster']].copy()

            dd['Cluster'] = dd['Cluster'].apply(lambda x: str(x))

            fig2 = ff.create_scatterplotmatrix(dd,
                                              diag='box', index ='Cluster',)

            fig2.update_layout(
                autosize = False,
                width = 800,
                height = 1250,
            )
            fig2.update_layout( title_text = '<b style = "font-family: Arial, sans-serif">Metric Correlations & Distributions</b>', 
                              title_font = dict(size = 30, family = 'cooper black'),
                               title = dict(x = 0.5, xanchor = 'center')
                
            )
        except Exception as e:
            st.write(f'An error occured: {e}')

        st.plotly_chart(fig2)

        st.markdown(
            """
            <div style="border-color: #d3d3d3; border-width: 4px; border-style: solid; border-radius: 10px; padding: 15px; margin: 10px; font-family: Arial, Helvetica, sans-serif; box-shadow: 5px 5px 10px grey;">
              <h3 style="color: gray; margin-bottom: 10px;"><b>A Detailed Exploration of Channel Categories 🔍</b></h3>
              
              <h5 style="color: blue;"><b>Category 1: Rising Stars💫</b></h5> 
              <p>In this category, channels emerge with fewer visits, likes, and subscribers, steadily climbing the ranks of Top YouTubers. They embody the aspiring talents, on the verge of breaking into the mainstream.</p>
              
              <h5 style="color: #FFA500;"><b>Category 2: Ground Zero📉</b></h5> 
              <p>This category represents the YouTubers on the lowest end of the spectrum. Among the top YouTubers, they fall behind the most in all the respective metrics, and are also very populous.</p>
              
              <h5 style="color: #008000;"><b>Category 3: Subscribers' Haven✨</b></h5> 
              <p>This category hosts channels with a substantial subscriber base but modest likes and visits. Recognized for their high retention rates, they craft popular content, albeit at a less frequent pace, resulting in a distinct engagement pattern.</p>
              
              <h5 style="color: #FF0000;"><b>Category 4: Balancing Act 🦾</b></h5> 
              <p>Moderate in visits, likes, and subscribers, these channels strike a balance on the lower spectrum, outshining <b style="color: #FFA500;">Category 2</b> in overall metrics. They hold a middle ground, contributing to the diverse YouTube landscape.</p>
              
              <h5 style="color: #800080;"><b>Category 5: Engaging Echoes🔊</b></h5> 
              
              <p style = "color:default;">Channels in this category boast the highest likes and visits, yet maintain a humble subscriber count. They epitomize high engagement but wrestle with retention rates, creating a vibrant but fleeting viewership.</p>
            </div>

            <p></p>

            """,
            unsafe_allow_html=True
        )


        #Cluster concentrations in Different Continents

        st.write('##### _**How are the Clusters Distributed by Continent?**_ 🌍')

        try:
            clusters = data['Cluster'].unique()

            color_sets = [px.colors.sequential.Magma, px.colors.sequential.Cividis, px.colors.sequential.Viridis, px.colors.diverging.Spectral]

            for cluster in clusters:
                data_ = data.loc[data['Cluster'] == cluster]

                country_dist = data_['Country'].value_counts().reset_index()


                fig4 = px.pie(country_dist, values = 'count', names = 'Country',
                            color_discrete_sequence = color_sets[cluster % len(color_sets)])

                fig4.update_layout(title_text='<b style = "font-family: Arial, sans-serif">Country Distribution in Cluster {}</b>'.format(cluster),
                            title_font=dict(size=30, family='Arial'),
                                title = dict(x = 0.5, xanchor = 'center'))

                fig4.update_layout(autosize = True)

                st.plotly_chart(fig4)

        except Exception as e:
            st.write(f'An error occured: {e}')


        st.title("**Other Important Metrics**")
        with st.expander('**Expand to see Continent Preferrences Globally**'):
            try:
                dataforplot = data['Categories'].value_counts().reset_index()
                dataforplot['Categories'] = dataforplot['Categories'].str.replace('Salud y autoayuda', 'Health and self-help')
                fig6 = px.pie(dataforplot, values = 'count', names = 'Categories', color_discrete_sequence = px.colors.diverging.PRGn)
                fig6.update_layout(title_text='<b style = "font-family: Arial, sans-serif;">Category Distributions</b>',
                              title_font=dict(size=30, family='Arial'),
                                    title = dict(x = 0.5, xanchor = 'center'))
                fig6.update_layout(autosize = False,
                    width = 800, height = 700)

                st.plotly_chart(fig6)

            except Exception as e:
                st.write(f'Error: {e}')




        with st.expander("**Expand to see Which content are preferred across Continents** 🌐"):
            try:
                color_sets = [px.colors.sequential.Magma, px.colors.sequential.Cividis, px.colors.sequential.Viridis]
                continents = data['Continent'].unique().tolist()
                for continent in continents:
                    if continent == 'Unknown':
                        continue
                    relevant_data = data.loc[data.Continent == continent]
                    relevant_data['Categories'] = relevant_data['Categories'].str.replace('Salud y autoayuda', 'Health and self-help')
                    cluster_proportions = relevant_data['Categories'].value_counts().reset_index()

                    fig7 = px.bar(cluster_proportions, x = 'count', y = 'Categories', color_discrete_sequence = color_sets[continents.index(continent) % len(color_sets)])

                    fig7.update_layout(title_text=f'<b style = "font-family: Arial, sans-serif">Categories in {continent}</b>',
                                title_font=dict(size=25, family='Arial'),
                                title=dict(x=0.5, xanchor='center'),
                                                autosize=False, width = 700, height = 550)

                    st.plotly_chart(fig7)

            except Exception as e:
                st.write(e)

    if selected == 'Top YouTubers in Categories':
        st.markdown(
        """
        <style>
            .channel-name {
                color: gray; 
            }
            .rounded-images {
                border-radius: 15px;
                box-shadow: 0 5px 10px rgba(0, 0, 0, 0.5);
                overflow: hidden;
                margin-bottom: 20px;
            }

            div {
                font-family: Arial, sans-serif;
            }

        </style>

        <div>
            <h2 class="channel-name"><b>PewDiePie & Mr Beast (Category 3)</b></h2>
            <p>PewDiePie is a Swedish YouTuber who is known for his gaming videos, comedy sketches, and meme reviews. He is <b>the most-subscribed individual creator</b> on YouTube with over <b>110 million subscribers</b>. Mr Beast is an American YouTuber who is famous for his expensive stunts, philanthropic acts, and viral challenges. He has over <b>80 million subscribers</b> and is one of the highest-earning YouTubers in the world.</p>
            <div class="rounded-images">
                <img src="https://hips.hearstapps.com/hmg-prod/images/pewdiepie_gettyimages-501661286.jpg?resize=1200:*" alt="PewDiePie" width="350", height = "350">
                <img src="https://wallpapers.com/images/hd/mr-beast-bright-screen-background-he6y102ildr4ca8q.jpg" alt="Mr Beast" width="349", height = "350">
            </div>
            <p></p><p></p>
        </div>

        <div>
            <h2 class="channel-name"><b>The Ellen Show & Katy Perry (Category 4)</b></h2>
            <p>The Ellen Show is an American daytime television variety comedy talk show hosted by Ellen DeGeneres. It has been running for <b>19 seasons</b> and has won numerous awards, including 11 Daytime Emmys for Outstanding Talk Show Entertainment. Katy Perry is an American singer, songwriter, and television personality. She is one of the best-selling music artists of all time, with over <b>143 million records sold worldwide</b>. She has nine U.S. number one singles and has received various accolades, including five American Music Awards and a Brit Award</p>
            <div class="rounded-images">
                <img src="https://m.media-amazon.com/images/M/MV5BODA5ZDQyMzYtZWQwMy00MDQ1LWE2OGUtNGYyNTk0Y2NhZGM4XkEyXkFqcGdeQXVyMTkzODUwNzk@._V1_.jpg" alt="The Ellen Show" width="350" height = "450">
                <img src="https://m.media-amazon.com/images/M/MV5BMjE4MDI3NDI2Nl5BMl5BanBnXkFtZTcwNjE5OTQwOA@@._V1_.jpg" alt="Katy Perry" width="349" height = "450">
            </div>
            <p></p><p></p>
        </div>

        <div>
            <h2 class="channel-name"><b>Techno Gamers & Kimberly Loaiza (Category 5)</b></h2>
            <p>Techno Gamers is an Indian gaming YouTuber who creates videos of gameplays and live streams of <b>GTA 5</b>, <b>Minecraft</b>, <b>Call of Duty</b>, and more. He has <b>over 37 million subscribers</b> and is one of the most popular gamers in India. Kimberly Loaiza is a Mexican internet personality and singer who started her YouTube career in 2016. She is currently the seventh most-followed user on TikTok and has over <b>40 million subscribers</b> on YouTube. She also has a music career and has released several singles, such as <em><b>Enamorarme</b>, <b>Patán</b></em>, and <em><b>Kitty</b></em>.</p>
            <div class="rounded-images">
                <img src="https://img.gurugamer.com/resize/740x-/2021/04/02/youtuber-ujjwal-techno-gamerz-3aa0.jpg" alt="Techno Gamerz" width="350" height = "450">
                <img src="https://m.media-amazon.com/images/I/71G48FB73WL._AC_UF1000,1000_QL80_.jpg" alt="Kimberly Loaiza" width="349" height = "450">
            </div>
            <p></p><p></p>
        </div>

        <div>
            <h2 class="channel-name"><b>SSSniperWolf & JackSepticEye (Category 1)</b></h2>
            <p>SSSniperWolf is a British-American YouTuber who is known for her gaming and reaction videos. She has over <b>30 million subscribers</b> and is one of the most-watched female gamers on YouTube. JackSepticEye is an Irish YouTuber who is also known for his gaming and vlog videos. He has over <b>27 million subscribers</b> and is one of the most influential Irish online personalities. He has also appeared in the film Free Guy and released a biographical documentary called <b><em>How Did We Get Here?</em></b></p>
            <div class="rounded-images">
                <img src="https://ih1.redbubble.net/image.2189561281.9428/mwo,x1000,ipad_2_skin-pad,750x1000,f8f8f8.u1.jpg" alt="SSSniper Wolf" width="350" height = "420">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Jacksepticeye_by_Gage_Skidmore.jpg/1200px-Jacksepticeye_by_Gage_Skidmore.jpg" alt="JackSepticEye" width="349" height = "420">
            </div>
            <p></p><p></p>
        </div>

        <div>
            <h2 class="channel-name"><b>JessNoLimit & Daddy Yankee (Category 2)</b></h2>
            <p>JessNoLimit is an Indonesian gaming YouTuber and Instagram star who is known for his Mobile Legends gameplays. He has over <b>42 million subscribers</b> and is the <b>third most-subscribed YouTuber in Indonesia</b>. Daddy Yankee is a Puerto Rican rapper, singer, songwriter, and actor who is considered the <b><em>"King of Reggaeton"</em></b>. He has sold over <b>30 million records worldwide</b> and has won numerous awards, including five Latin Grammy Awards and two Billboard Music Awards. He is best known for his hit songs like <em><b>Gasolina</b>, <b>Despacito</b></em>, and <em><b>Con Calma</b></em>.</p>
            <div class="rounded-images">
                <img src="https://akcdn.detik.net.id/visual/2023/05/05/jess-no-limit-dan-sisca-kohl-2_43.png?w=650&q=90" alt="JessNoLimit" width="350" height = "300">
                <img src="https://people.com/thmb/eT6A-wncUzuDs-XV08qRSd_gSUk=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(688x281:690x283)/Daddy-Yankee-Retirement-120623-a855484297944821ad14c8b98453b6a5.jpg" alt="Daddy Yankee" width="349" height = "300">
            </div>
        </div>

        """,
        unsafe_allow_html=True
        )
        st.write("_** Ranked based on Popularity_")

    if selected == 'ML Highlights / About Project':
        st.title(
            """
            **The Magic Behind the Scenes ✨**
            
            ###### _(How We Trained and Validated Our Model)_
            ---

            """
        )

        st.markdown(
            """
            <div style = "font-family: Arial, sans-serif;"><p>You might be curious about how we're generating the awesome recommendations for you. Well, the secret is <b>a powerful machine learning model</b> from the <code>Scikit-Learn</code> library, and with some clever hyperparameter-tuning and other techniques, we achieved some amazing results! 🙌</p>
            <p>Here are some highlights from the training process:</p><div>
            """, unsafe_allow_html = True
        )

        st.write('---')


        col1, col2 = st.columns([.5, .5])
        st.write('---')
        col1.write("#### Training & Evaluation Model:")
        col2.write(RandomForestClassifier(n_estimators = 100, max_features = 0.5, min_samples_split = 2, bootstrap = True))
        col1.write('**Some Techniques used:**')
        col1.write('`GridSearchCV()`, `cross_val_score()`, `mutual_information()`, `PCA()`, `KMeans() CLustering`')
        rf_model = score_model(data = data, model = RandomForestClassifier(), model_params = model_params)


        st.write("_**A glace at the dataset:**_")

        table = st.write(data.head(10))

        st.write("See the full dataset [here](https://github.com/zahemen9900/YouTube-Analytics-App/blob/main/YouTube%20Data%20EDA/yt_cluster_data.csv)")

        st.markdown(
            """
            <p></p>
            <div style="font-size: 15px; font-family: Arial, sans-serif">
            <h5><b>More on Data Used</b></h5>

            <p>As an aspiring Top YouTuber, you deserve to compare yourself with the best of the best. That's why we used a curated dataset of the world's top 1000 YouTubers, to give you accurate assessments and useful tips to level up your game. We hope you enjoyed the recommendation as much as we enjoyed making this project 😊</p>

            <p>If you have any suggestions or recommendations, feel free to leave them in the <b>side-bar to your left ↖️</b>, and I'd really appreciate it</p>

            <p>If you are curious about how we made this app possible, or want to explore more resources for the project, you can check out <b>Zahemen's GitHub</b> <a href = "https://github.com/zahemen9900/YouTube-Analytics-App">here</a> and <a href = "https://github.com/zahemen9900/Analytics_App_YT">here</a> . You will find the source code, the data, and more.</p>
            <p>Leave a star on my GitHub if you enjoyed using the app, and thank you for your time and attention!</p></div>

            """,
            unsafe_allow_html=True
        )



    if selected == 'Your Recommendations':

        if 'recs' not in st.session_state:
            st.session_state['recs'] = None


        if 'recommendations' not in st.session_state:
            st.session_state['recommendations'] = None

        st.header("Enter your channel link below")

        with st.form(key = 'channel-url-form'):
            yt_url = st.text_input(label = '**Channel URL goes here** _( please remove any quotation marks from URL)_')

            st.write("Not sure how to get the link? Visit [this page](https://www.wikihow.com/Find-Your-YouTube-URL)")


            st.write("Try any of these as examples (just copy into link section):")

            st.write({
                'channel_url': 'www.youtube.com/channel/UCBJycsmduvYEL83R_U4JriQ',
                'category': 'Technology'
            })

            '_**or**_'
            
            st.write({
                'channel_url': 'https://www.youtube.com/channel/UCv8G-xZ_BBGufjbN6jbvLMQ',
                'category': 'Comedy'
            })

            selected_cat = st.radio("#### **Choose your channel category**", (
                    "Animation", "Toys", "Movies", "Video Games", "Music and Dance",
                    "News and Politics", "Fashion", "Animals and Pets", "Education",
                    "DIY and Life Hacks", "Fitness", "ASMR", "Comedy", "Technology", "Automobiles",
                    "Food and Cooking", "Travel and Adventure", "Science and Technology",
                    "Health and Wellness", "History and Documentaries", "Lifestyle Content",
                    "Book Reviews", "Art and Creativity", "Home Decor", "Parenting", "Business and Finance",
                    "Social Issues", "Photography", "Spirituality", "Language Learning",
                    "Sports and Athletics", "Entertainment News", "Pop Culture", "Podcasts"))

            submit_btn = st.form_submit_button('Submit', help = 'Submit to see your channel ')

            if submit_btn:
                if not (yt_url.startswith('https://www.youtube.com/channel/') or yt_url.startswith('www.youtube.com/channel/') or yt_url.startswith('https://youtube.com/channel/') or yt_url.startswith('youtube.com/channel/')):
                    st.error('Please enter a valid channel URL. Also make sure your link does not contain quotation marks')
                    st.session_state['recs'] = None
                else:
                    st.success('Url received!')
                    with st.spinner('Retrieving channel info...'):
                        time.sleep(1.5)

                    new_channel_df = extract_channel_info(yt_url, selected_cat)
                    if isinstance(new_channel_df, str):
                        st.session_state['recs'] = None
                        
                    with st.spinner('Getting your recommendations...'):
                        time.sleep(1.5)

                    recs = generate_recommendations(df = data, yt_channel_df = new_channel_df, model = st.session_state['rf_model_main'])
                    
                    st.write(recs)
                    st.session_state['recs'] = recs
        
        if st.session_state['recs'] is not None:
            rc_ = deliver_recommendations('form_auto', st.session_state['recs'])



        st.write("### Or enter your details manually below")


        with st.expander("**Expand to enter your info manually**"):
            if "formbtn_state" not in st.session_state:
                st.session_state.formbtn_state = False

            if st.session_state.formbtn_state:
                st.session_state.formbtn_state = True

            with st.form(key="channel_form"):

                channel_name = st.text_input('What is your channel name?', 'eg. zahemen9900')
                st.write("Select the category of your videos:")
                selected_category = st.radio("Options", (
                    "Animation", "Toys", "Movies", "Video Games", "Music and Dance",
                    "News and Politics", "Fashion", "Animals and Pets", "Education",
                    "DIY and Life Hacks", "Fitness", "ASMR", "Comedy", "Technology", "Automobiles",
                    "Food and Cooking", "Travel and Adventure", "Science and Technology",
                    "Health and Wellness", "History and Documentaries", "Lifestyle Content",
                    "Book Reviews", "Art and Creativity", "Home Decor", "Parenting", "Business and Finance",
                    "Social Issues", "Photography", "Spirituality", "Language Learning",
                    "Sports and Athletics", "Entertainment News", "Pop Culture", "Podcasts"
                ))

                selected_country = st.selectbox("Select your country", popular_countries)

                default_continent = continent_mapping.get(selected_country, 'Unknown')

                continents = data['Continent'].unique()
                continents = [continent for continent in continents]

                # Get the index of the default continent in the list of continents
                # If the default continent is not in the list, use 0 as the index
                default_index = continents.index(default_continent) if default_continent in continents else 0

                selected_continent = st.selectbox("Select your Continent", continents, index = default_index)



                n_visits = st.text_input('How many Visits do you have on average?')

                n_likes = st.text_input('How many likes do you get on average?')

                n_subs = st.text_input('How many subscribers do you have?')

                submit_button = st.form_submit_button(label="Submit", help = 'Submit to see your recommendations!')

                if submit_button and not any(metric is None for metric in [n_visits, n_likes, n_subs]):
                    try:
                        n_visits, n_likes, n_subs = int(n_visits), int(n_likes), int(n_subs)
                    except:
                        st.error("Couldn't proceed. Please make sure that visits, likes and subscribers are all entered and are all numeric")
                        submit_button = False

                if submit_button:
                    st.success('Form submitted, Results are underway!')
                    time.sleep(1)

                    with st.spinner('Loading recomendations'):
                        time.sleep(1)

                    args =  pd.DataFrame({'Username' : channel_name,
                            'Categories': selected_category,
                            'Subscribers': n_subs,
                            'Country': selected_country,
                            'Continent' : selected_continent,
                            'Visits' : n_visits,
                            'Likes' : n_likes
                            }, index = [0])

                    personalized = generate_recommendations(df = data, model = st.session_state['rf_model_main'], yt_channel_df=args)


                    st.write(personalized)

                    st.session_state['recommendations'] = personalized

        if st.session_state['recommendations'] is not None:
            rc = deliver_recommendations('form_manual', st.session_state['recommendations'])



if __name__ == '__main__':
    main()
