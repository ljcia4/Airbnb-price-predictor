import pandas as pd
import os
from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(base_dir, 'csv/raw_Airbnb_Napoli.csv')
output_path = os.path.join(base_dir, 'csv/clean_Airbnb_Napoli.csv')
train_output_path = os.path.join(base_dir, 'csv/train_Airbnb_Napoli.csv')
test_output_path = os.path.join(base_dir, 'csv/test_Airbnb_Napoli.csv')

def clean_dataset():
    df = pd.read_csv(dataset_path, low_memory=False)
    
    # Rimozione colonne non rilevanti
    columns_to_drop = [
        'id', 'listing_url', 'scrape_id', 'last_scraped', 'source', 
        'picture_url', 'host_id', 'host_url', 'host_name', 
        'host_thumbnail_url', 'host_picture_url', 'bathrooms_text', 
        'calendar_last_scraped', 'license', 'host_neighbourhood',
        'name', 'description', 'neighborhood_overview', 'host_since', 
        'host_location', 'host_about', 'host_verifications', 'neighbourhood', 
        'neighbourhood_group_cleansed', 'property_type',
        'minimum_minimum_nights', 'maximum_minimum_nights', 
        'minimum_maximum_nights', 'maximum_maximum_nights', 'calendar_updated',
        'availability_eoy', 'number_of_reviews_ly',
        'review_scores_accuracy', 'review_scores_cleanliness', 
        'review_scores_checkin', 'review_scores_communication',
        'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 
        'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms'
    ]
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

    # Pulizia Specifica del prezzo
    if 'price' in df_cleaned.columns:
        df_cleaned['price'] = df_cleaned['price'].astype(str).replace({r'\$': '', ',': ''}, regex=True)
        df_cleaned['price'] = pd.to_numeric(df_cleaned['price'], errors='coerce')
        
        df_cleaned = df_cleaned.dropna(subset=['price'])


    # Divisione Train/Test
    train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)
             
    # Salvataggio
    files_map = {
        output_path: df_cleaned,
        train_output_path: train_df,
        test_output_path: test_df
    }

    for path, data in files_map.items():
        data.to_csv(path, index=False)

if __name__ == "__main__":
    clean_dataset()