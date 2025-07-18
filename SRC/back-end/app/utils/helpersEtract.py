import json 
import os
import requests #type: ignore
from datetime import datetime


# ====================================================== Helepers Functions for data extraction =========================================================

def get_data(api_url):
    try:
        response = requests.get(
            api_url,
            headers={'Content-Type': 'application/json'}
        )
        
        response.raise_for_status()

        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

def append_to_json_file(data, file):
    if os.path.exists(file):
        return 
    
    with open(file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"Data saved to {file}")


def fetch_and_concatenate_data(apis):
    all_data = []
    for api_url in apis:
        data = get_data(api_url)
        if data:  
            all_data.extend(data)
        else:
            print(f"Failed to fetch data from {api_url}")
    return all_data


def load_json_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. The file might be corrupted or not in valid JSON format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}")
        return None
    
def store_guest_nationalities(reservations):
    """
    Extracts unique guest nationalities from a list of reservations.
    """
    guest_nationalities_set = set()
    
    for reservation in reservations:
        if "guestNationality" in reservation and reservation["guestNationality"]:
            guest_nationalities_set.add(reservation["guestNationality"])
    
    return guest_nationalities_set


def store_customer_countries(reservations):
    """
    Extracts unique customer countries from a list of reservations.
    """
    countries_set = set()
    
    for reservation in reservations:
        if "customerCountry" in reservation and reservation["customerCountry"]:
            countries_set.add(reservation["customerCountry"])
    
    return countries_set


def get_avr_of_sejour(actives, client=None):
    """
    Calculates the average length of stay for all reservations or for a specific client.
    """
    if not actives:
        return 0
        
    if client:
        reservations_to_use = [reservation for reservation in actives if reservation.get("customer") == client["name"]]
    else:
        reservations_to_use = actives
    
    if not reservations_to_use:
        return 0
        
    total_nights = sum(reservation.get("nights", 0) or 0 for reservation in reservations_to_use)
    return total_nights / len(reservations_to_use)


def convert_date_fr_to_iso(date_fr):
    """
    Converts a date from French format (DD-MM-YYYY) to ISO format (YYYY-MM-DD).
    """
    if isinstance(date_fr, str) and len(date_fr.split("-")) == 3:
        day, month, year = date_fr.split("-")
        return f"{year}-{month}-{day}"
    return date_fr


def compter_reservations_actives_par_dates(reservations, dates, total_rooms, client=None):
    """
    Counts active reservations by date and calculates various metrics.
    """
    
    lignes = [
        'Total rooms', 'Room nights', 'Occupancy percentage', 'Bed nights',
        'adults', 'kids', 'Available rooms', 'Revenue', 'IF',
        'ADR Nte', 'ADR (Average daily rate)', 'Revpar'
    ]
    
    resultats = {ligne: {} for ligne in lignes}
    line_totals = {ligne: 0 for ligne in lignes}
    
    for date_donnee in dates:
        date_cible = datetime.fromisoformat(date_donnee).date()
        
        reservations_actives = []
        for reservation in reservations:
            date_debut = datetime.fromisoformat(convert_date_fr_to_iso(reservation["checkin"])).date()
            date_fin = datetime.fromisoformat(convert_date_fr_to_iso(reservation["checkout"])).date()
            
            if client:
                if date_cible >= date_debut and date_cible < date_fin and reservation.get("customer") == client["name"]:
                    reservations_actives.append(reservation)
            else:
                if date_cible >= date_debut and date_cible < date_fin:
                    reservations_actives.append(reservation)
        
        for ligne in lignes:
            resultats[ligne][date_donnee] = calcul(reservations_actives, ligne, total_rooms, line_totals)
    
    return resultats, line_totals


def calcul(reservations_actives, ligne, total_rooms, line_totals):
    """
    Calculates a specific metric for a list of active reservations.
    """
    if ligne == 'Total rooms':
        line_totals[ligne] += total_rooms
        return total_rooms
        
    elif ligne == 'Room nights':
        room_nights = len(reservations_actives)
        line_totals[ligne] += room_nights
        return room_nights
        
    elif ligne == 'Occupancy percentage':
        room_nights = len(reservations_actives)
        percentage = (room_nights / total_rooms) * 100 if total_rooms > 0 else 0
        line_totals[ligne] = (line_totals['Room nights'] / line_totals['Total rooms']) * 100 if line_totals['Total rooms'] > 0 else 0
        return round(percentage, 2)
        
    elif ligne == 'Bed nights':
        bed_nights = sum((res.get("adult", 0) or 0) + (res.get("child", 0) or 0) + (res.get("infant", 0) or 0) 
                         for res in reservations_actives)
        line_totals[ligne] += bed_nights
        return bed_nights
        
    elif ligne == 'adults':
        adults = sum(res.get("adult", 0) or 0 for res in reservations_actives)
        line_totals[ligne] += adults
        return adults
        
    elif ligne == 'kids':
        kids = sum((res.get("child", 0) or 0) + (res.get("infant", 0) or 0) for res in reservations_actives)
        line_totals[ligne] += kids
        return kids
        
    elif ligne == 'Available rooms':
        available = total_rooms - len(reservations_actives)
        line_totals[ligne] = line_totals['Total rooms'] - line_totals['Room nights']
        return available
        
    elif ligne == 'Revenue':
        revenue = sum((res.get("localCurrencyPrice", 0) or 0) / (res.get("nights", 1) or 1) for res in reservations_actives)
        line_totals[ligne] += revenue
        return round(revenue, 2)
        
    elif ligne == 'IF':
        bed_nights = sum((res.get("adult", 0) or 0) + (res.get("child", 0) or 0) + (res.get("infant", 0) or 0) 
                         for res in reservations_actives)
        room_nights = len(reservations_actives)
        if_value = bed_nights / room_nights if room_nights > 0 else 0
        line_totals[ligne] = line_totals['Bed nights'] / line_totals['Room nights'] if line_totals['Room nights'] > 0 else 0
        return round(if_value, 2)
        
    elif ligne == 'ADR Nte':
        bed_nights = sum((res.get("adult", 0) or 0) + (res.get("child", 0) or 0) + (res.get("infant", 0) or 0) 
                         for res in reservations_actives)
        revenue = sum((res.get("localCurrencyPrice", 0) or 0) / (res.get("nights", 1) or 1) for res in reservations_actives)
        adr_nte = revenue / bed_nights if bed_nights > 0 else 0
        line_totals[ligne] = line_totals['Revenue'] / line_totals['Bed nights'] if line_totals['Bed nights'] > 0 else 0
        return round(adr_nte, 2)
        
    elif ligne == 'ADR (Average daily rate)':
        room_nights = len(reservations_actives)
        revenue = sum((res.get("localCurrencyPrice", 0) or 0) / (res.get("nights", 1) or 1) for res in reservations_actives)
        adr = revenue / room_nights if room_nights > 0 else 0
        line_totals[ligne] = line_totals['Revenue'] / line_totals['Room nights'] if line_totals['Room nights'] > 0 else 0
        return round(adr, 2)
        
    elif ligne == 'Revpar':
        revenue = sum((res.get("localCurrencyPrice", 0) or 0) / (res.get("nights", 1) or 1) for res in reservations_actives)
        revpar = revenue / total_rooms if total_rooms > 0 else 0
        line_totals[ligne] = line_totals['Revenue'] / line_totals['Total rooms'] if line_totals['Total rooms'] > 0 else 0
        return round(revpar, 2)
    
    return 0


def get_dates(from_date, to_date):
    from datetime import datetime, timedelta
    
    start_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.strptime(to_date, "%Y-%m-%d")
    
    dates = []
    current_date = start_date
    
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    return dates


def get_occupancy_data(start_date_str, end_date_str, total_rooms_param):
    """
    Fetches reservation data for a given date range, calculates occupancy,
    and returns the occupancy data per date and line totals.
    """
    api_templates = [
        'https://pmsvaleriaapi.fractalstay.com/api/reservations?&from={}&to={}&group=1&resastatus=0',
        'https://pmsvaleriaapi.fractalstay.com/api/reservations?&from={}&to={}&group=1&resastatus=3',
        'https://pmsvaleriaapi.fractalstay.com/api/reservations?&from={}&to={}&group=1&resastatus=2',
    ]

    apis_for_date_range = [template.format(start_date_str, end_date_str) for template in api_templates]
    print(f"Sending requests to the following APIs: {len(apis_for_date_range)}")

    reservation_data = fetch_and_concatenate_data(apis_for_date_range)

    if not reservation_data:
        print("No data fetched. Cannot calculate occupancy.")
        return None
    
    print(f"Successfully fetched {len(reservation_data)} reservations.")

    dates_in_range = get_dates(start_date_str, end_date_str)
    print(f"Calculating occupancy for dates: {len(dates_in_range)}")
    
    occupancy_results, line_totals_results = compter_reservations_actives_par_dates(reservation_data, dates_in_range, total_rooms_param)
    print("Occupancy calculation complete.")
    
    return {
        "occupancy_par_date": occupancy_results,
        "line_totals": line_totals_results
    }