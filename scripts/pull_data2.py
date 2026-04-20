import pandas as pd
from googleapiclient.discovery import build

API_KEY = "nothin"
VIDEO_ID = "6GSIxaeES5I"

def fetch_and_save_raw_data(video_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response['items']:
            snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'commenterID': snippet['authorChannelId']['value'], # Unique ID
                'commentBody': snippet['textDisplay'],
                'date_posted': snippet['publishedAt']
            })

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    df = pd.DataFrame(comments)
    df.to_csv("iran_war.csv", index=False)
    print(f"Saved {len(df)} comments to iran_war.csv")

if __name__ == "__main__":
    fetch_and_save_raw_data(VIDEO_ID, API_KEY)