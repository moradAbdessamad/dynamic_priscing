import json 
import os
import requests #type: ignore
from datetime import datetime
import os
import json
import requests #type: ignore
from dotenv import load_dotenv #type: ignore
from collections import defaultdict
from typing import List, Dict,DefaultDict
import datetime
from dateutil.parser import parse #type: ignore
from typing import List, Dict, Any, Optional, Set
import re
from datetime import timedelta #type: ignore

load_dotenv()
AUTH_URL = os.getenv("AUTH_URL")
USERNAME = os.getenv("USERNAME")
SCOPE = os.getenv("SCOPE")
GRANT_TYPE = os.getenv("GRANT_TYPE")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
PASSWORD = os.getenv("PASSWORD")

def get_the_access(auth_url=AUTH_URL,
                   username=USERNAME,
                   scope=SCOPE,
                   grant_type=GRANT_TYPE,
                   client_id=CLIENT_ID,
                   client_secret=CLIENT_SECRET,
                   password=PASSWORD):
    """
    Get the access token from the API using the client credentials.
    """
    
    auth_payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "username": username,
        "password": password,
        "grant_type": grant_type,
        "scope": scope
    }

    token_response = requests.post(auth_url, data=auth_payload)

    if token_response.status_code == 200:
        token_data = token_response.json()
        access_token = token_data["access_token"]
        print("Token obtained successfully!")
        return access_token
    
    else:
        print(f"Error {token_response.status_code}: {token_response.text}")
        return None
    

def append_to_json_file(data, file):
    if os.path.exists(file):
        return 
    
    with open(file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"Data saved to {file}")


def load_json_data(file_path):
    """
    Load JSON data from a given file path.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


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


def calcul(reservations_actives, ligne, total_rooms, line_totals):
    """
    Calculates a specific metric for a list of active reservations.
    Only calculates: Total rooms, Room nights (needed for occupancy), 
    Occupancy percentage, adults, kids, Available rooms, and Revenue
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
    
    return 0


def get_all_rooms(api_url):
    """
    Get all rooms from the API.
    
    Args:
        api_url (str): The base API URL
    """

    access_token = get_the_access()
    
    if not access_token:
        print("Failed to obtain access token")
        return None
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    group_id = 1  
    
    url = f"{api_url}/rooms/findAll?&group={group_id}"
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print("Rooms retrieved successfully!")
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None
    

def get_room_types_from_rooms(rooms):
    """
    Extract room types from rooms data and calculate total non-virtual rooms.
    
    Args:
        rooms (List[Dict]): List of room dictionaries
    """
    if not rooms:
        print("No rooms data provided")
        return [], {}, 0

    room_types = []
    room_type_counts = {}
    total_non_virtual = 0
    
    for room in rooms:
        if not room.get('isVirtual', False):
            total_non_virtual += 1

        room_type = room.get('roomType')
        if not room_type or not room_type.get('name'):
            continue
        
        room_type_name = room_type['name'].strip()
        
        if room_type_name not in room_types:
            room_types.append(room_type_name)
        
        room_type_counts[room_type_name] = room_type_counts.get(room_type_name, 0) + 1
    
    return room_types, room_type_counts, total_non_virtual


def convert_date_fr_to_iso(date_fr):
    """
    Converts a date from French format (DD-MM-YYYY) to ISO format (YYYY-MM-DD).
    """
    if isinstance(date_fr, str) and len(date_fr.split("-")) == 3:
        day, month, year = date_fr.split("-")
        return f"{year}-{month}-{day}"
    return date_fr


def compter_reservations_par_dates(reservations, dates, client=None):
    """
    Counts active reservations by date and returns the actual reservation objects
    grouped by date.
    """
    
    reservations_par_date = {}
    
    for date_donnee in dates:
        date_cible = datetime.date.fromisoformat(date_donnee) 
        
        reservations_actives = []
        for reservation in reservations:
            checkin_str = reservation.get("checkin")
            checkout_str = reservation.get("checkout")

            if not checkin_str or not checkout_str:
                continue 

            try:
                date_debut = datetime.date.fromisoformat(convert_date_fr_to_iso(checkin_str))
                date_fin = datetime.date.fromisoformat(convert_date_fr_to_iso(checkout_str)) 
            except ValueError:
                continue 
            
            if client:
                if date_cible >= date_debut and date_cible < date_fin and reservation.get("customer") == client.get("name"): 
                    reservations_actives.append(reservation)
            else:
                if date_cible >= date_debut and date_cible < date_fin:
                    reservations_actives.append(reservation)
        
        reservations_par_date[date_donnee] = reservations_actives
    
    return reservations_par_date


def compter_reservations_par_roomtype(reservations, 
                                     dates, 
                                     target_room_type_names, 
                                     client = None):
    """
    Groups active reservations by date and predefined room types.
    Returns a dictionary where:
    - Each date is a key.
    - The value is a dictionary containing:
        - "room_types": A dictionary where keys are room type names (from target_room_type_names and "Unknown")
                        and values are lists of reservations for that room type.
        - "counts": A dictionary where keys are room type names (from target_room_type_names and "Unknown")
                    and values are the counts of reservations for that room type.
        - "total_reservations": Total number of active reservations for the date.
    """
    
    data_par_date: Dict[str, Dict[str, Any]] = {}
    
    target_room_types_set = set(target_room_type_names)
    
    all_categories_to_track = list(target_room_type_names)
    if "Unknown" not in target_room_types_set: 
        all_categories_to_track.append("Unknown")

    for date_donnee in dates:
        date_cible = datetime.date.fromisoformat(date_donnee) 
        
        room_types_for_date: Dict[str, List[Dict[str, Any]]] = {rt_name: [] for rt_name in all_categories_to_track}
        counts_for_date: Dict[str, int] = {rt_name: 0 for rt_name in all_categories_to_track}
        
        reservations_actives_for_date: List[Dict[str, Any]] = []
        for reservation in reservations:
            try:
                checkin_str = reservation.get("checkin")
                checkout_str = reservation.get("checkout")

                if not checkin_str or not checkout_str:
                    continue 

                date_debut = datetime.date.fromisoformat(convert_date_fr_to_iso(checkin_str)) 
                date_fin = datetime.date.fromisoformat(convert_date_fr_to_iso(checkout_str)) 
            except (ValueError, TypeError):
                continue 

            is_active_on_date = (date_cible >= date_debut and date_cible < date_fin)
            
            if client:
                if is_active_on_date and client.get("name") and reservation.get("customer") == client.get("name"):
                    reservations_actives_for_date.append(reservation)
            else:
                if is_active_on_date:
                    reservations_actives_for_date.append(reservation)
        
        for reservation in reservations_actives_for_date:
            raw_room_type_info = reservation.get("roomType")
            extracted_room_type_name = "Unknown" 

            if isinstance(raw_room_type_info, dict):
                name_val = raw_room_type_info.get("name")
                
                if name_val and isinstance(name_val, str):
                    extracted_room_type_name = name_val.strip()
                
                elif name_val: 
                    extracted_room_type_name = str(name_val).strip()
            
            elif isinstance(raw_room_type_info, str):
                extracted_room_type_name = raw_room_type_info.strip()
            
            if not extracted_room_type_name: 
                extracted_room_type_name = "Unknown"

            if extracted_room_type_name in target_room_types_set:
                room_types_for_date[extracted_room_type_name].append(reservation)
                counts_for_date[extracted_room_type_name] += 1
            else:
                room_types_for_date["Unknown"].append(reservation)
                counts_for_date["Unknown"] += 1
        
        data_par_date[date_donnee] = {
            "room_types": room_types_for_date,
            "counts": counts_for_date,
            "total_reservations": len(reservations_actives_for_date)
        }
    
    return data_par_date


def merge_room_types(data, merge_map):
    """
    Merges room types across multiple dates based on a mapping.

    Args:
        data (dict): Nested dictionary with dates as keys and "room_types" as subkeys.
        merge_map (dict): Mapping of old room type names to new room type names.
    """
    for date, day_data in data.items():
        original_room_types = day_data.get("room_types", {})
        new_room_types = {}

        for room_type, room_list in original_room_types.items():
            normalized_type = merge_map.get(room_type, room_type)

            if normalized_type not in new_room_types:
                new_room_types[normalized_type] = []

            new_room_types[normalized_type].extend(room_list)

        day_data["room_types"] = new_room_types

    return data


def merge_room_type_names(room_types, room_type_counts, merge_map):
    """
    Merge room types and update their counts based on a merge map.

    Args:
        room_types (list): List of original room type names.
        room_type_counts (dict): Dictionary mapping room type to its count.
        merge_map (dict): Mapping from old names to new unified names.
    """
    merged_counts = {}

    for room_type in room_types:
        normalized_type = merge_map.get(room_type, room_type)

        count = room_type_counts.get(room_type, 0)

        if normalized_type not in merged_counts:
            merged_counts[normalized_type] = 0

        merged_counts[normalized_type] += count

    updated_room_types = list(merged_counts.keys())

    return updated_room_types, merged_counts


def get_dates(from_date, to_date):
    from datetime import datetime, timedelta
    
    start_date_obj = datetime.strptime(from_date, "%Y-%m-%d") 
    end_date_obj = datetime.strptime(to_date, "%Y-%m-%d")   
    
    dates = []
    current_date_obj = start_date_obj 
    
    while current_date_obj <= end_date_obj:
        dates.append(current_date_obj.strftime("%Y-%m-%d"))
        current_date_obj += timedelta(days=1)
    
    return dates


def compter_reservations_actives_par_dates(reservations, dates, total_rooms, client=None):
    """
    Counts active reservations by date and calculates selected metrics.
    """
    
    lignes = [
        'Total rooms', 'Room nights', 'Occupancy percentage', 
        'adults', 'kids', 'Available rooms', 'Revenue'
    ]
    
    resultats = {ligne: {} for ligne in lignes}
    line_totals = {ligne: 0 for ligne in lignes} 
    
    for date_donnee in dates:
        date_cible = datetime.date.fromisoformat(date_donnee) 
        
        reservations_actives = []
        for reservation in reservations:
            try:
                checkin_str = reservation.get("checkin")
                checkout_str = reservation.get("checkout")
                if not checkin_str or not checkout_str:
                    continue
                date_debut = datetime.date.fromisoformat(convert_date_fr_to_iso(checkin_str)) 
                date_fin = datetime.date.fromisoformat(convert_date_fr_to_iso(checkout_str)) 
            except (ValueError, TypeError):
                continue 

            if client:
                if date_cible >= date_debut and date_cible < date_fin and reservation.get("customer") == client.get("name"):
                    reservations_actives.append(reservation)
            else:
                if date_cible >= date_debut and date_cible < date_fin:
                    reservations_actives.append(reservation)
        
        for ligne in lignes:
            resultats[ligne][date_donnee] = calcul(reservations_actives, ligne, total_rooms, line_totals)
    
    return resultats, line_totals


def calculate_metrics_per_roomtype_per_date(
    data_per_roomtype, 
    dates, 
    room_type_inventory, 
    target_room_type_names):
    """
    Calculates metrics for each specified room type on each date.

    Args:
        data_per_roomtype: Data structured as {date: {"room_types": {room_type_name: [reservations]}}}.
                           This comes from the modified compter_reservations_par_roomtype function.
        dates: List of date strings ("YYYY-MM-DD").
        room_type_inventory: Dictionary mapping room type names to their total counts (e.g., {'Standard Room': 50}).
        target_room_type_names: List of room type names to process.
    
    Returns:
        A tuple: (metrics_by_date_then_roomtype, overall_line_totals_by_roomtype)
        - metrics_by_date_then_roomtype: {date: {room_type: {metric: value}}}
        - overall_line_totals_by_roomtype: {room_type: {metric: total_value_over_period}}
    """
    lignes = [
        'Total rooms', 'Room nights', 'Occupancy percentage', 
        'adults', 'kids', 'Available rooms', 'Revenue'
    ]
    
    metrics_by_date_then_roomtype: Dict[str, Dict[str, Dict[str, Any]]] = {date_str: {} for date_str in dates}
    overall_line_totals_by_roomtype: Dict[str, Dict[str, float]] = {}

    for room_type_name in target_room_type_names:
        if room_type_name == "Unknown": 
            continue
        
        inventory_for_type = room_type_inventory.get(room_type_name, 0)
        if inventory_for_type == 0:
            print(f"Skipping room type '{room_type_name}' due to zero inventory.")
            continue
        
        overall_line_totals_by_roomtype[room_type_name] = {ligne: 0.0 for ligne in lignes}

    for date_donnee in dates:
        metrics_by_date_then_roomtype[date_donnee] = {}
        
        for room_type_name in target_room_type_names:
            if room_type_name == "Unknown": 
                continue
            
            if room_type_name not in overall_line_totals_by_roomtype: 
                continue

            current_room_type_total_rooms = room_type_inventory.get(room_type_name, 0)
            
            reservations_actives_for_type_and_date = []
            if date_donnee in data_per_roomtype and \
               "room_types" in data_per_roomtype[date_donnee] and \
               room_type_name in data_per_roomtype[date_donnee]["room_types"]:
                reservations_actives_for_type_and_date = data_per_roomtype[date_donnee]["room_types"][room_type_name]

            metrics_by_date_then_roomtype[date_donnee][room_type_name] = {}
            
            current_room_type_overall_totals = overall_line_totals_by_roomtype[room_type_name]

            for ligne in lignes:
                calculated_value = calcul(
                    reservations_actives_for_type_and_date, 
                    ligne, 
                    current_room_type_total_rooms, 
                    current_room_type_overall_totals 
                )
                metrics_by_date_then_roomtype[date_donnee][room_type_name][ligne] = calculated_value

    return metrics_by_date_then_roomtype, overall_line_totals_by_roomtype


def filter_reservations_by_customer(reservations, customer_target="COMPLI-Gratuit"):
    """
    Filters a list of reservations, removing entries where the 'customer'
    matches the customer_target.

    Args:
        reservations: A list of reservation dictionaries.
        customer_target: The customer name to filter out.
    """
    if not customer_target:
        return reservations, []

    kept_reservations: List[Dict[str, Any]] = []
    removed_reservations_list: List[Dict[str, Any]] = []

    for reservation in reservations:
        customer_name = reservation.get("customer")
        if customer_name == customer_target:
            removed_reservations_list.append(reservation)
        else:
            kept_reservations.append(reservation)
            
    return kept_reservations, removed_reservations_list


def run_full_reservation_analysis_units_pipeline(
    start_date_str,
    end_date_str,
    initial_room_types,
    initial_room_type_counts,
    api_templates,
    customer_target_to_filter="COMPLI-Gratuit"):
    """
    Runs the full pipeline for fetching, processing, and analyzing reservation data.
    Allows filtering reservations by a specific customer and adjusting inventory counts.
    
    Args:
        start_date_str: The start date in "YYYY-MM-DD" format.
        end_date_str: The end date in "YYYY-MM-DD" format.
        initial_room_types: List of unique room type names.
        initial_room_type_counts: Dictionary mapping room type names to their total inventory counts.
        customer_target_to_filter: Optional customer name to filter out from reservations.
                                   If provided, inventory counts will also be adjusted.
    """

    print("Step 1: API templates defined.")

    apis_for_date_range = [template.format(start_date_str, end_date_str) for template in api_templates]
    print(f"\nStep 2: Sending requests to {len(apis_for_date_range)} APIs for date range {start_date_str} to {end_date_str}.")
    reservation_data = fetch_and_concatenate_data(apis_for_date_range)

    if not reservation_data:
        print("Error: No reservation data fetched. Aborting pipeline.")
        return None, None
    print(f"Successfully fetched {len(reservation_data)} original reservations.")

    current_reservation_data = reservation_data
    adjusted_initial_room_type_counts = initial_room_type_counts.copy() 

    if customer_target_to_filter:
        print(f"\nStep 2.1: Filtering reservations for customer: '{customer_target_to_filter}' and adjusting inventory.")
        current_reservation_data, removed_reservations = filter_reservations_by_customer(
            reservation_data, 
            customer_target_to_filter
        )
        print(f"  {len(removed_reservations)} reservations removed for this customer.")
        print(f"  {len(current_reservation_data)} reservations remaining for analysis.")

        if removed_reservations:
            removed_counts_by_room_type = defaultdict(int)

            for res in removed_reservations:
                raw_room_type_info = res.get("roomType")
                room_type_name_in_res = "Unknown" 

                if isinstance(raw_room_type_info, dict):
                    name_val = raw_room_type_info.get("name")

                    if name_val and isinstance(name_val, str):
                        room_type_name_in_res = name_val.strip()

                    elif name_val:
                        room_type_name_in_res = str(name_val).strip()

                elif isinstance(raw_room_type_info, str):
                    room_type_name_in_res = raw_room_type_info.strip()
                
                if not room_type_name_in_res: 
                    room_type_name_in_res = "Unknown"
                
                removed_counts_by_room_type[room_type_name_in_res] += 1
            
            print(f"  Adjusting room type inventory based on {len(removed_reservations)} removed reservations:")
            for rt_name, num_removed in removed_counts_by_room_type.items():

                if rt_name in adjusted_initial_room_type_counts:
                    original_count = adjusted_initial_room_type_counts[rt_name]
                    adjusted_initial_room_type_counts[rt_name] = max(0, original_count - num_removed)

                    if original_count != adjusted_initial_room_type_counts[rt_name]:
                         print(f"    Room Type '{rt_name}': inventory {original_count} -> {adjusted_initial_room_type_counts[rt_name]} (-{num_removed})")
                else:
                    print(f"    Warning: Room Type '{rt_name}' (from {num_removed} removed reservations) not found in initial inventory keys. Its count not directly adjusted in this step.")
    else:
        print("\nStep 2.1: No customer target specified for filtering. Using all fetched reservations and original inventory counts.")


    dates_in_range_list = get_dates(start_date_str, end_date_str)
    if not dates_in_range_list:
        print("Error: Could not generate dates for the given range. Aborting pipeline.")
        return None, None
    print(f"\nStep 3: Processing for {len(dates_in_range_list)} dates from {start_date_str} to {end_date_str}.")

    print("\nStep 4: Grouping reservations by room type per date...")
    data_per_roomtype = compter_reservations_par_roomtype(
        current_reservation_data, 
        dates_in_range_list,
        initial_room_types 
    )
    print("Reservations grouped by room type per date.")

    merge_map = {
        "Chambre Standard": "Standard Room",
    }
    print(f"\nStep 5: Merging room type reservation lists within daily data using map: {merge_map}")
    merged_data = merge_room_types(data_per_roomtype, merge_map) 
    print("Room type reservation lists merged within daily data.")

    print("\nStep 6: Generating new list of room types and their inventory counts after merging names...")

    new_room_types, new_room_type_counts = merge_room_type_names(
        initial_room_types,
        adjusted_initial_room_type_counts, 
        merge_map
    )
    print(f"New room types (post-merge): {new_room_types}")
    print(f"New room type inventory counts (post-merge and adjustment): {new_room_type_counts}")

    print("\nStep 7: Calculating detailed metrics per merged room type per date...")
    detailed_room_type_metrics, overall_room_type_totals = calculate_metrics_per_roomtype_per_date(
        merged_data, 
        dates_in_range_list,
        new_room_type_counts,    
        new_room_types           
    )
    print("Detailed metrics calculation complete.")
    
    print("Pipeline execution finished.")
    return detailed_room_type_metrics, overall_room_type_totals


def extract_and_prepare_reservation_data(
    start_date_str,
    end_date_str,
    initial_room_types,
    api_templates,
    initial_room_type_counts,
    customer_target_to_filter="COMPLI-Gratuit"):
    """
    Fetches, filters, groups, and merges reservation data without calculating final metrics.
    
    Args:
        start_date_str: The start date in "YYYY-MM-DD" format.
        end_date_str: The end date in "YYYY-MM-DD" format.
        initial_room_types: List of unique room type names.
        initial_room_type_counts: Dictionary mapping room type names to their total inventory counts.
        customer_target_to_filter: Optional customer name to filter out from reservations.
                                   If provided, inventory counts will also be adjusted.
    """

    print("Step 1: API templates defined.")

    apis_for_date_range = [template.format(start_date_str, end_date_str) for template in api_templates]
    print(f"\nStep 2: Sending requests to {len(apis_for_date_range)} APIs for date range {start_date_str} to {end_date_str}.")
    reservation_data = fetch_and_concatenate_data(apis_for_date_range)

    if not reservation_data:
        print("Error: No reservation data fetched. Aborting processing.")
        return None, None, None, None
    print(f"Successfully fetched {len(reservation_data)} original reservations.")

    current_reservation_data = reservation_data
    adjusted_initial_room_type_counts = initial_room_type_counts.copy() 

    if customer_target_to_filter:
        print(f"\nStep 2.1: Filtering reservations for customer: '{customer_target_to_filter}' and adjusting inventory.")
        current_reservation_data, removed_reservations = filter_reservations_by_customer(
            reservation_data, 
            customer_target_to_filter
        )
        print(f"  {len(removed_reservations)} reservations removed for this customer.")
        print(f"  {len(current_reservation_data)} reservations remaining for analysis.")

        if removed_reservations:
            removed_counts_by_room_type: DefaultDict[str, int] = defaultdict(int)
            for res in removed_reservations:
                raw_room_type_info = res.get("roomType")
                room_type_name_in_res = "Unknown" 
                if isinstance(raw_room_type_info, dict):
                    name_val = raw_room_type_info.get("name")
                    if name_val and isinstance(name_val, str):
                        room_type_name_in_res = name_val.strip()
                    elif name_val:
                        room_type_name_in_res = str(name_val).strip()
                elif isinstance(raw_room_type_info, str):
                    room_type_name_in_res = raw_room_type_info.strip()
                if not room_type_name_in_res: 
                    room_type_name_in_res = "Unknown"
                removed_counts_by_room_type[room_type_name_in_res] += 1
            
            print(f"  Adjusting room type inventory based on {len(removed_reservations)} removed reservations:")
            for rt_name, num_removed in removed_counts_by_room_type.items():
                if rt_name in adjusted_initial_room_type_counts:
                    original_count = adjusted_initial_room_type_counts[rt_name]
                    adjusted_initial_room_type_counts[rt_name] = max(0, original_count - num_removed)
                    
                    if original_count != adjusted_initial_room_type_counts[rt_name]:
                         print(f"    Room Type '{rt_name}': inventory {original_count} -> {adjusted_initial_room_type_counts[rt_name]} (-{num_removed})")
                
                else:
                    print(f"    Warning: Room Type '{rt_name}' (from {num_removed} removed reservations) not found in initial inventory keys. Its count not directly adjusted in this step.")
    
    else:
        print("\nStep 2.1: No customer target specified for filtering. Using all fetched reservations and original inventory counts.")

    dates_in_range_list = get_dates(start_date_str, end_date_str)
    if not dates_in_range_list:
        print("Error: Could not generate dates for the given range. Aborting processing.")
        return None, None, None, None
    print(f"\nStep 3: Processing for {len(dates_in_range_list)} dates from {start_date_str} to {end_date_str}.")

    print("\nStep 4: Grouping reservations by room type per date...")
    data_per_roomtype = compter_reservations_par_roomtype(
        current_reservation_data, 
        dates_in_range_list,
        initial_room_types 
    )
    print("Reservations grouped by room type per date.")

    merge_map = {
        "Chambre Standard": "Standard Room",
    }
    print(f"\nStep 5: Merging room type reservation lists within daily data using map: {merge_map}")
    merged_data = merge_room_types(data_per_roomtype, merge_map) 
    print("Room type reservation lists merged within daily data.")

    print("\nStep 6: Generating new list of room types and their inventory counts after merging names...")
    new_room_types, new_room_type_counts = merge_room_type_names(
        initial_room_types,
        adjusted_initial_room_type_counts, 
        merge_map
    )
    print(f"New room types (post-merge): {new_room_types}")
    print(f"New room type inventory counts (post-merge and adjustment): {new_room_type_counts}")
    
    print("Data extraction and preparation finished.")
    return merged_data, new_room_types, new_room_type_counts, dates_in_range_list


def structure_prepared_data_for_json(merged_data, new_room_types, new_room_type_counts, dates_in_range_list):
    """
    Structures the prepared reservation data into a single dictionary format
    suitable for JSON serialization.

    Args:
        merged_data: Processed reservation data grouped by date and merged room type.
        new_room_types: List of room type names after merging.
        new_room_type_counts: Dictionary of room type inventory counts after merging and adjustment.
        dates_in_range_list: List of dates in the specified range.
    """
    
    structured_output = {
        "processing_date_range": dates_in_range_list,
        "room_type_inventory_summary": {
            "room_types_list": new_room_types,
            "room_type_counts": new_room_type_counts
        },
        "daily_room_reservations": merged_data
    }
    
    return structured_output


def run_analysis_in_intervals(
    start_date_str,
    end_date_str,
    initial_room_types,
    initial_room_type_counts,
    api_templates,
    interval_days=7,
    customer_target_to_filter="COMPLI-Gratuit"
):
    """
    Run the reservation analysis pipeline in intervals and aggregate the results.
    
    Args:
        start_date_str: Overall start date in "YYYY-MM-DD" format.
        end_date_str: Overall end date in "YYYY-MM-DD" format.
        initial_room_types: List of unique room type names.
        initial_room_type_counts: Dictionary mapping room type names to their inventory counts.
        interval_days: Number of days for each analysis interval.
        api_templates: List of API templates to use for fetching reservation data.
        customer_target_to_filter: Optional customer name to filter out from reservations.
    """

    from datetime import datetime as dt
    start_date = dt.strptime(start_date_str, "%Y-%m-%d")
    end_date = dt.strptime(end_date_str, "%Y-%m-%d")
    
    aggregated_detailed_metrics = {}
    aggregated_overall_totals = {}
    for room_type in initial_room_types:
        aggregated_overall_totals[room_type] = {
            'Total rooms': 0,
            'Room nights': 0,
            'Occupancy percentage': 0,
            'adults': 0,
            'kids': 0,
            'Available rooms': 0,
            'Revenue': 0
        }
    
    current_start = start_date
    interval_count = 0
    
    while current_start < end_date:

        current_end = min(current_start + timedelta(days=interval_days-1), end_date)
        
        current_start_str = current_start.strftime("%Y-%m-%d")
        current_end_str = current_end.strftime("%Y-%m-%d")
        
        print(f"\n=== Processing interval {interval_count+1}: {current_start_str} to {current_end_str} ===")
        
        detailed_metrics, overall_totals = run_full_reservation_analysis_units_pipeline(
            current_start_str,
            current_end_str,
            initial_room_types,
            initial_room_type_counts,
            api_templates,
            customer_target_to_filter
        )
        
        if not detailed_metrics or not overall_totals:
            print(f"Warning: No results for interval {current_start_str} to {current_end_str}")
            current_start = current_end + timedelta(days=1)
            interval_count += 1
            continue
        
        for date_str, room_type_data in detailed_metrics.items():
            aggregated_detailed_metrics[date_str] = room_type_data
        
        for room_type, metrics in overall_totals.items():
            if room_type not in aggregated_overall_totals:
                aggregated_overall_totals[room_type] = metrics
            else:
                for metric, value in metrics.items():
                    if metric == 'Occupancy percentage':
                        continue
                    aggregated_overall_totals[room_type][metric] += value
        
        current_start = current_end + timedelta(days=1)
        interval_count += 1
    
    for room_type, metrics in aggregated_overall_totals.items():
        total_rooms = metrics['Total rooms']
        room_nights = metrics['Room nights']
        if total_rooms > 0:
            metrics['Occupancy percentage'] = round((room_nights / total_rooms) * 100, 2)
        else:
            metrics['Occupancy percentage'] = 0
    
    print(f"\n=== Completed analysis across {interval_count} intervals ===")
    print(f"Total date range: {start_date_str} to {end_date_str}")
    
    return aggregated_detailed_metrics, aggregated_overall_totals




# ========================================================== The function for segments ==========================================================

def fetch_and_filter_reservations_by_date_chunked(
    start_date_str,
    end_date_str,
    api_templates,
    customer_target_to_filter="COMPLI-Gratuit",
    client=None,
    chunk_days=5
):
    """
    Fetches reservation data in chunks to handle API limitations, filters by customer, 
    and groups reservations by date.
    
    Args:
        start_date_str: Start date in "YYYY-MM-DD" format.
        end_date_str: End date in "YYYY-MM-DD" format.
        api_templates: List of API URL templates.
        customer_target_to_filter: Customer name to filter out from reservations.
        client: Optional client filter for final date grouping.
        chunk_days: Maximum number of days per API request (default: 5).
    
    Returns:
        tuple: (reservations_par_date, dates_list, summary_stats)
    """
    
    from datetime import datetime, timedelta
    
    print(f"Starting chunked data fetch for range: {start_date_str} to {end_date_str}")
    print(f"Chunk size: {chunk_days} days")
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    all_reservation_data = []
    chunk_details = []
    chunk_count = 0
    
    current_start = start_date
    while current_start <= end_date:
        chunk_count += 1
        
        chunk_end = min(current_start + timedelta(days=chunk_days - 1), end_date)
        
        chunk_start_str = current_start.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")
        
        print(f"\n--- Processing Chunk {chunk_count}: {chunk_start_str} to {chunk_end_str} ---")
        
        apis_for_chunk = [template.format(chunk_start_str, chunk_end_str) for template in api_templates]
        print(f"Sending requests to {len(apis_for_chunk)} APIs for chunk {chunk_count}")
        
        chunk_data = fetch_and_concatenate_data(apis_for_chunk)
        
        if chunk_data:
            print(f"Chunk {chunk_count}: Fetched {len(chunk_data)} reservations")
            all_reservation_data.extend(chunk_data)
            
            chunk_details.append({
                "chunk": chunk_count,
                "start": chunk_start_str,
                "end": chunk_end_str,
                "reservations": len(chunk_data)
            })
        else:
            print(f"Warning: No data returned for chunk {chunk_count}")
            chunk_details.append({
                "chunk": chunk_count,
                "start": chunk_start_str,
                "end": chunk_end_str,
                "reservations": 0
            })
        
        current_start = chunk_end + timedelta(days=1)
    
    if not all_reservation_data:
        print("Error: No reservation data fetched from any chunk. Aborting processing.")
        return None, None, None
    
    print(f"\nTotal reservations fetched across {chunk_count} chunks: {len(all_reservation_data)}")
    
    current_reservation_data = all_reservation_data
    removed_reservations = []
    
    if customer_target_to_filter:
        print(f"\nFiltering reservations for customer: '{customer_target_to_filter}'")
        current_reservation_data, removed_reservations = filter_reservations_by_customer(
            all_reservation_data, 
            customer_target_to_filter
        )
        print(f"  {len(removed_reservations)} reservations removed for this customer.")
        print(f"  {len(current_reservation_data)} reservations remaining for analysis.")
    else:
        print("\nNo customer target specified for filtering. Using all fetched reservations.")
    
    dates_list = get_dates(start_date_str, end_date_str)
    if not dates_list:
        print("Error: Could not generate dates for the given range. Aborting processing.")
        return None, None, None
    
    print(f"\nProcessing for {len(dates_list)} dates from {start_date_str} to {end_date_str}.")
    
    print("Grouping reservations by date...")
    reservations_par_date = compter_reservations_par_dates(
        current_reservation_data, 
        dates_list, 
        client
    )
    print("Reservations grouped by date.")
    
    summary_stats = {
        "total_chunks_processed": chunk_count,
        "total_reservations_fetched": len(all_reservation_data),
        "total_reservations_filtered": len(current_reservation_data),
        "total_removed_reservations": len(removed_reservations),
        "date_range": {
            "start": start_date_str,
            "end": end_date_str,
            "total_days": len(dates_list)
        },
        "chunk_details": chunk_details,
        "customer_filter_applied": customer_target_to_filter,
        "client_filter_applied": client.get("name") if client else None
    }
    
    print(f"\nChunked processing completed successfully!")
    print(f"Summary: {summary_stats['total_reservations_fetched']} total → {summary_stats['total_reservations_filtered']} filtered")
    
    return reservations_par_date, dates_list, summary_stats


def classify_by_segment_and_count(data_by_date):
    """
    Classifies reservations by customerSegment per date,
    and returns total counts per date.
    """
    classified = {}
    counts = {}

    for date, reservations in data_by_date.items():
        segment_groups = defaultdict(list)

        for res in reservations:
            segment = res.get("customerSegment", "UNKNOWN")
            segment_groups[segment].append(res)

        classified[date] = dict(segment_groups)

        counts[date] = {segment: len(res_list) for segment, res_list in segment_groups.items()}

    return classified, counts







from datetime import datetime, date, timedelta #type: ignore

def getAllSegments(base_url, token, timeout):
    """
    Fetch the list of market-segments that exist in the PMS / CRM.
    """
    url = f"{base_url.rstrip('/')}/segments"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()                

    return response.json()    


def get_segment_names(segments_data):
    """
    Extract segment names from the segments API response data.
    
    Args:
        segments_data: Dictionary containing the API response with embedded segments
    """
    try:
        segments = segments_data['_embedded']['segments']
        return [segment['name'] for segment in segments]
    except KeyError as e:
        print(f"Error accessing data structure: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []
    


def compter_reservations_par_segment(reservations, 
                                   dates, 
                                   target_segments, 
                                   client=None):
    """
    Groups active reservations by date and predefined customer segments.
    Similar to compter_reservations_par_roomtype but for segments.
    """
    from datetime import datetime, timedelta
    
    data_par_date: Dict[str, Dict[str, Any]] = {}
    
    target_segments_set = set(target_segments)
    
    all_categories_to_track = list(target_segments)
    if "UNKNOWN" not in target_segments_set: 
        all_categories_to_track.append("UNKNOWN")

    for date_donnee in dates:
        date_cible = datetime.fromisoformat(date_donnee).date()
        
        segments_for_date: Dict[str, List[Dict[str, Any]]] = {segment_name: [] for segment_name in all_categories_to_track}
        counts_for_date: Dict[str, int] = {segment_name: 0 for segment_name in all_categories_to_track}
        
        reservations_actives_for_date: List[Dict[str, Any]] = []
        for reservation in reservations:
            try:
                checkin_str = reservation.get("checkin")
                checkout_str = reservation.get("checkout")

                if not checkin_str or not checkout_str:
                    continue 

                date_debut = datetime.fromisoformat(convert_date_fr_to_iso(checkin_str)).date()
                date_fin = datetime.fromisoformat(convert_date_fr_to_iso(checkout_str)).date()
            except (ValueError, TypeError):
                continue 

            is_active_on_date = (date_cible >= date_debut and date_cible < date_fin)
            
            if client:
                if is_active_on_date and client.get("name") and reservation.get("customer") == client.get("name"):
                    reservations_actives_for_date.append(reservation)
            else:
                if is_active_on_date:
                    reservations_actives_for_date.append(reservation)
        
        for reservation in reservations_actives_for_date:
            segment_value = reservation.get("customerSegment", "UNKNOWN")
            
            if not segment_value or segment_value is None:
                segment_value = "UNKNOWN"
            
            if segment_value in target_segments_set:
                segments_for_date[segment_value].append(reservation)
                counts_for_date[segment_value] += 1
            else:
                segments_for_date["UNKNOWN"].append(reservation)
                counts_for_date["UNKNOWN"] += 1
        
        data_par_date[date_donnee] = {
            "segments": segments_for_date,
            "counts": counts_for_date,
            "total_reservations": len(reservations_actives_for_date)
        }
    
    return data_par_date


def calculate_metrics_per_segment_per_date(
    data_per_segment, 
    dates, 
    total_hotel_rooms, 
    target_segments):
    """
    Calculates metrics for each specified customer segment on each date.
    Similar to calculate_metrics_per_roomtype_per_date but for segments.
    """
    lignes = [
        'Total rooms', 'Room nights', 'Occupancy percentage', 
        'adults', 'kids', 'Available rooms', 'Revenue'
    ]
    
    metrics_by_date_then_segment: Dict[str, Dict[str, Dict[str, Any]]] = {date_str: {} for date_str in dates}
    overall_line_totals_by_segment: Dict[str, Dict[str, float]] = {}

    for segment_name in target_segments:
        if segment_name == "UNKNOWN": 
            continue
        overall_line_totals_by_segment[segment_name] = {ligne: 0.0 for ligne in lignes}

    for date_donnee in dates:
        metrics_by_date_then_segment[date_donnee] = {}
        
        for segment_name in target_segments:
            if segment_name == "UNKNOWN": 
                continue
            
            if segment_name not in overall_line_totals_by_segment: 
                continue

            reservations_actives_for_segment_and_date = []
            if date_donnee in data_per_segment and \
               "segments" in data_per_segment[date_donnee] and \
               segment_name in data_per_segment[date_donnee]["segments"]:
                reservations_actives_for_segment_and_date = data_per_segment[date_donnee]["segments"][segment_name]

            metrics_by_date_then_segment[date_donnee][segment_name] = {}
            
            current_segment_overall_totals = overall_line_totals_by_segment[segment_name]

            for ligne in lignes:
                calculated_value = calcul(
                    reservations_actives_for_segment_and_date, 
                    ligne, 
                    total_hotel_rooms, 
                    current_segment_overall_totals 
                )
                metrics_by_date_then_segment[date_donnee][segment_name][ligne] = calculated_value

    return metrics_by_date_then_segment, overall_line_totals_by_segment


def run_full_reservation_analysis_segments_pipeline(
    start_date_str,
    end_date_str,
    target_segments,
    total_hotel_rooms,
    api_templates,
    customer_target_to_filter="COMPLI-Gratuit"):
    """
    Runs the full pipeline for segment analysis instead of room type analysis.
    """

    print("Step 1: API templates defined for segment analysis.")

    apis_for_date_range = [template.format(start_date_str, end_date_str) for template in api_templates]
    print(f"\nStep 2: Sending requests to {len(apis_for_date_range)} APIs for date range {start_date_str} to {end_date_str}.")
    reservation_data = fetch_and_concatenate_data(apis_for_date_range)

    if not reservation_data:
        print("Error: No reservation data fetched. Aborting pipeline.")
        return None, None
    print(f"Successfully fetched {len(reservation_data)} original reservations.")

    current_reservation_data = reservation_data

    if customer_target_to_filter:
        print(f"\nStep 2.1: Filtering reservations for customer: '{customer_target_to_filter}'.")
        current_reservation_data, removed_reservations = filter_reservations_by_customer(
            reservation_data, 
            customer_target_to_filter
        )
        print(f"  {len(removed_reservations)} reservations removed for this customer.")
        print(f"  {len(current_reservation_data)} reservations remaining for analysis.")
    else:
        print("\nStep 2.1: No customer target specified for filtering. Using all fetched reservations.")

    dates_in_range_list = get_dates(start_date_str, end_date_str)
    if not dates_in_range_list:
        print("Error: Could not generate dates for the given range. Aborting pipeline.")
        return None, None
    print(f"\nStep 3: Processing for {len(dates_in_range_list)} dates from {start_date_str} to {end_date_str}.")

    print("\nStep 4: Grouping reservations by customer segment per date...")
    data_per_segment = compter_reservations_par_segment(
        current_reservation_data, 
        dates_in_range_list,
        target_segments 
    )
    print("Reservations grouped by customer segment per date.")

    print("\nStep 5: Calculating detailed metrics per customer segment per date...")
    detailed_segment_metrics, overall_segment_totals = calculate_metrics_per_segment_per_date(
        data_per_segment, 
        dates_in_range_list,
        total_hotel_rooms,    
        target_segments           
    )
    print("Detailed segment metrics calculation complete.")
    
    print("Segment analysis pipeline execution finished.")
    return detailed_segment_metrics, overall_segment_totals


def run_segment_analysis_in_intervals(
    start_date_str,
    end_date_str,
    target_segments,
    total_hotel_rooms,
    api_templates,
    interval_days=7,
    customer_target_to_filter="COMPLI-Gratuit"
):
    """
    Run the segment analysis pipeline in intervals and aggregate the results.
    """

    from datetime import datetime as dt
    start_date = dt.strptime(start_date_str, "%Y-%m-%d")
    end_date = dt.strptime(end_date_str, "%Y-%m-%d")
    
    aggregated_detailed_metrics = {}
    aggregated_overall_totals = {}
    for segment in target_segments:
        aggregated_overall_totals[segment] = {
            'Total rooms': 0,
            'Room nights': 0,
            'Occupancy percentage': 0,
            'adults': 0,
            'kids': 0,
            'Available rooms': 0,
            'Revenue': 0
        }
    
    current_start = start_date
    interval_count = 0
    
    while current_start < end_date:

        current_end = min(current_start + timedelta(days=interval_days-1), end_date)
        
        current_start_str = current_start.strftime("%Y-%m-%d")
        current_end_str = current_end.strftime("%Y-%m-%d")
        
        print(f"\n=== Processing segment interval {interval_count+1}: {current_start_str} to {current_end_str} ===")
        
        detailed_metrics, overall_totals = run_full_reservation_analysis_segments_pipeline(
            current_start_str,
            current_end_str,
            target_segments,
            total_hotel_rooms,
            api_templates,
            customer_target_to_filter
        )
        
        if not detailed_metrics or not overall_totals:
            print(f"Warning: No results for interval {current_start_str} to {current_end_str}")
            current_start = current_end + timedelta(days=1)
            interval_count += 1
            continue
        
        for date_str, segment_data in detailed_metrics.items():
            aggregated_detailed_metrics[date_str] = segment_data
        
        for segment, metrics in overall_totals.items():
            if segment not in aggregated_overall_totals:
                aggregated_overall_totals[segment] = metrics
            else:
                for metric, value in metrics.items():
                    if metric == 'Occupancy percentage':
                        continue
                    aggregated_overall_totals[segment][metric] += value
        
        current_start = current_end + timedelta(days=1)
        interval_count += 1
    
    for segment, metrics in aggregated_overall_totals.items():
        total_rooms = metrics['Total rooms']
        room_nights = metrics['Room nights']
        if total_rooms > 0:
            metrics['Occupancy percentage'] = round((room_nights / total_rooms) * 100, 2)
        else:
            metrics['Occupancy percentage'] = 0
    
    print(f"\n=== Completed segment analysis across {interval_count} intervals ===")
    print(f"Total date range: {start_date_str} to {end_date_str}")
    
    return aggregated_detailed_metrics, aggregated_overall_totals