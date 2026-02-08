import pandas as pd
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
        'neighbourhood_cleansed','amenities'
    ]
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

    # Pulizia Specifica del prezzo
    if 'price' in df_cleaned.columns:
        df_cleaned['price'] = df_cleaned['price'].astype(str).replace({r'\$': '', ',': ''}, regex=True)
        df_cleaned['price'] = pd.to_numeric(df_cleaned['price'], errors='coerce')
        
        df_cleaned = df_cleaned.dropna(subset=['price'])

    # Gestione last_review
    if 'last_review' in df_cleaned.columns:
        df_cleaned['last_review'] = pd.to_datetime(df_cleaned['last_review'], errors='coerce')
        ref_date = df_cleaned['last_review'].max()
        df_cleaned['days_since_last_review'] = (ref_date - df_cleaned['last_review']).dt.days
        df_cleaned = df_cleaned.drop(columns=['last_review'])
        # Imputazione dei valori mancanti in days_since_last_review con la mediana
        df_cleaned['days_since_last_review'] = df_cleaned['days_since_last_review'].fillna(df_cleaned['days_since_last_review'].median())

    # Gestione host_response_rate e host_acceptance_rate
    rate_cols = ['host_response_rate', 'host_acceptance_rate']
    for col in rate_cols:
        if col in df_cleaned.columns:
            # Rimuove % e converte in numerico
            df_cleaned[col] = df_cleaned[col].astype(str).str.replace('%', '', regex=False)
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            
            # Imputazione con la mediana
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            
            # Binning
            bins = [0, 20, 40, 60, 80, 100]
            labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
            df_cleaned[col] = pd.cut(df_cleaned[col], bins=bins, labels=labels, include_lowest=True)
            
            # Convertire in stringa per assicurare corretta gestione come categorica
            df_cleaned[col] = df_cleaned[col].astype(str)

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