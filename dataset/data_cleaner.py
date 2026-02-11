import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

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
        'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms',
        'first_review', 'minimum_nights', 'maximum_nights',
        'neighbourhood_cleansed','amenities', 'host_response_rate', 'host_acceptance_rate', 'host_response_time', 'host_has_profile_pic'
    ]
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

    # Pulizia Specifica del prezzo
    if 'price' in df_cleaned.columns:
        df_cleaned['price'] = df_cleaned['price'].astype(str).replace({r'\$': '', ',': ''}, regex=True)
        df_cleaned['price'] = pd.to_numeric(df_cleaned['price'], errors='coerce')
        
        df_cleaned = df_cleaned.dropna(subset=['price'])

        # Rimozione outlier prezzo > 600
        df_cleaned = df_cleaned[df_cleaned['price'] <= 600]

    # Gestione last_review
    if 'last_review' in df_cleaned.columns:
        df_cleaned['last_review'] = pd.to_datetime(df_cleaned['last_review'], errors='coerce')
        ref_date = df_cleaned['last_review'].max()
        df_cleaned['days_since_last_review'] = (ref_date - df_cleaned['last_review']).dt.days
        df_cleaned = df_cleaned.drop(columns=['last_review'])
        # Imputazione dei valori mancanti in days_since_last_review con la mediana
        df_cleaned['days_since_last_review'] = df_cleaned['days_since_last_review'].fillna(df_cleaned['days_since_last_review'].median())

    # Calcolo distanza dal centro (Piazza Plebiscito)
    if 'latitude' in df_cleaned.columns and 'longitude' in df_cleaned.columns:
        center_lat = 40.8358256
        center_lon = 14.2485831
        
        df_cleaned['latitude'] = pd.to_numeric(df_cleaned['latitude'], errors='coerce')
        df_cleaned['longitude'] = pd.to_numeric(df_cleaned['longitude'], errors='coerce')
        
        # Formula di Haversine vettorizzata
        R = 6371  # Raggio della Terra in km
        phi1 = np.radians(df_cleaned['latitude'])
        phi2 = np.radians(center_lat)
        dphi = np.radians(center_lat - df_cleaned['latitude'])
        dlambda = np.radians(center_lon - df_cleaned['longitude'])
        
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        df_cleaned['distance_from_center'] = R * c
        
        # Rimuove le colonne originali
        df_cleaned = df_cleaned.drop(columns=['latitude', 'longitude'])

    # Imputazione Dati Mancanti
    
    # Variabili Categoriche: Moda
    cat_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if not df_cleaned[col].mode().empty:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

    # Variabili Numeriche: Mediana (tranne reviews_per_month)
    num_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if col == 'price':
            continue 
            
        if col in ['reviews_per_month', 'review_scores_rating', 'review_scores_location']:
             df_cleaned[col] = df_cleaned[col].fillna(0)
        else:
             df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # One-Hot Encoding per tutte le variabili rimaste
    df_encoded = pd.get_dummies(df_cleaned, drop_first=True, dtype=int)

    # Divisione Train/Test
    train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42)
             
    # Salvataggio
    files_map = {
        output_path: df_encoded,
        train_output_path: train_df,
        test_output_path: test_df
    }

    for path, data in files_map.items():
        data.to_csv(path, index=False)

if __name__ == "__main__":
    clean_dataset()