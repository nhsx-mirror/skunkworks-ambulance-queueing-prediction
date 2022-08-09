import datetime
from io import BytesIO, StringIO
import numpy as np
import pandas as pd
import requests
import shapely
import warnings

from acquisition import get_bank_holiday_data, get_weather_data


# Defining as global variable
HOSPITALS = ['hospital_A', 
            'hospital_B', 
            'hospital_C', 
            'hospital_D', 
            'hospital_E', 
            'hospital_F'
            ]  
            
    

def skills_by_crew_member(assignments_and_incidents_merged: pd.DataFrame):
    """
    This function outputs a boolean dataframe outlining the skills each crew member has.
    Parameters
    ----------
    assignments_and_incidents_merged : pd.DataFrame
        The fake assignments data and fake incidents data merged on the incident number.
    
    Returns
    -------
    pd.DataFrame of True/False depending on whether that crew member possesses that skill or not
    
    """
    
    df_skill = assignments_and_incidents_merged[['time', 
                                                 'hospital', 
                                                 'crew1_skill',
                                                 'crew2_skill',
                                                 'crew3_skill']]
    
    # Unpivoting crew1_skill, crew2_skill and crew3_skill into rows
    s = df_skill.melt(id_vars=['time', 'hospital'], var_name='crew', value_name='skill').set_index(['hospital', 'time', 'crew'])['skill'].dropna()
    
    # Split the crew skills where a comma appears and assigns it a True value, stacks arrays in sequence horizontally and fill missing values with NA
    df_skills = pd.DataFrame(np.hstack(s.str.split(',').apply(lambda words: {w.strip(): True for w in words})).tolist(), index=s.index).fillna(False)
    
    return df_skills



def past_events(cleaned_assignments: pd.DataFrame, cleaned_incidents: pd.DataFrame, cleaned_pts: pd.DataFrame):
    """
    Interleaves the fake assignments data, the fake incidents data and the fake PTS data.
    Parameters
    ----------
    cleaned_assignments : pd.DataFrame
        Fake assignments dataframe generated from fake data generator
    cleaned_incidents : pd.DataFrame
        Fake incidents dataframe generated from fake data generator
    cleaned_pts : pd.DataFrame
        Fake PTS dataframe generated from fake data generator
    
    Returns
    -------
    pd.DataFrame interleaving the three dataframes
    
    """
    
    # Merging the cleaned assignments and the cleaned incidents on the incident number
    df = cleaned_assignments.merge(cleaned_incidents, how='left', on='incident_number')
    
    # Calculating the number of ambulance arrivals by using time destination
    num_amb_arrivals = df.groupby(['hospital', 'time']).size().to_frame('num_ambulance_arrivals')

    # Calculating the number of ambulance leaving the queue by using time handover and time clear when HALOing occurs
    num_amb_depatures = df.groupby(['hospital', 'time_amb_left_queue']).size().to_frame('num_ambulance_depatures')
    num_amb_depatures = num_amb_depatures.reset_index().rename(columns={'time_amb_left_queue':'time'}).set_index(['hospital', 'time'])
    
    # Pivots data to get the number of people in each age band arriving at the hospital at each timestamp
    age_band = df[['time', 'hospital', 'age_band']]
    age_band = age_band.pivot_table(index=['hospital', 'time'], columns='age_band', aggfunc=len, fill_value=0)
    
    # Pivots data to get the number of people for each responding priority arriving at the hospital at each timestamp
    df['responding_priority'] = df['responding_priority'].astype('float')
    responding_priority = df[['time', 'hospital', 'responding_priority']]
    responding_priority = responding_priority.pivot_table(index=['hospital', 'time'], columns='responding_priority', aggfunc=len, fill_value=0)
    responding_priority = responding_priority.rename(columns={0.0: 'priority_0', 1.0: 'priority_1', 2.0: 'priority_2', 3.0: 'priority_3', 
                                                              4.0: 'priority_4', 5.0: 'priority_5', 6.0: 'priority_6', 7.0: 'priority_7', 
                                                              8.0: 'priority_8', 9.0: 'priority_9'}) #change value of field before pivot columns=lambnda d: priority_d int d
    
    # Pivots data to get the number of patients transported arriving at the hospital in each ambulance at each timestamp
    num_patients_transported = df[['time', 'hospital', 'num_patients_transported']]
    num_patients_transported = num_patients_transported.pivot_table(index=['hospital','time'], columns = 'num_patients_transported', aggfunc=len, fill_value=0)
    num_patients_transported = num_patients_transported.rename(columns={1: '1_patient_transported', 2: '2_patients_transported', 3: '3_patients_transported'})
    
    # Counts the number of ambulances that are missing each skill by hospital and time
    skills = skills_by_crew_member(df)
    missing_skills = ~skills
    missing_skills = missing_skills.groupby(['hospital', 'time'])\
        .all()\
        .astype(int)\
        .add_prefix('missing_')
    
    # Re-naming handover delay to past delay to use in subsequent function to calculate the average past delays over a 6 hour window
    past_delays = df[['hospital', 'time', 'handover_delay_mins']].rename(columns={'handover_delay_mins': 'past_delay_mins'})
    past_delays = past_delays.groupby(['hospital', 'time']).mean()
    
    # Joining all above dataframes to create one dataframe 
    past_events = num_amb_arrivals.join([missing_skills, cleaned_pts, past_delays, age_band, responding_priority, num_patients_transported, num_amb_depatures], how='outer')
    past_events = past_events.reset_index()
    
    return past_events



def prepare_delays(cleaned_assignments: pd.DataFrame):
    """
    Prepares a dataframe of describing the handover delay at each timestamp for the hospitals.
    Parameters
    ----------
    cleaned_assignments : pd.DataFrame
        Fake Assignments dataframe created by fake data generator
        
    Returns
    -------
    pd.DataFrame describing the handover delays
    """
    
    # Taking handover delays from cleaned assignments (arrived at emergency department at hospital) dataset
    delays = cleaned_assignments[['time', 'hospital', 'handover_delay_mins']]
    
    # Aim to predict at 3 hours, 10 hours, 24 hours, duplicated columns and re-named in order to merge in subsequet functions
    delays = delays.rename(columns={'time':'time_plus_3'})
    delays['time_plus_10'] = delays['time_plus_3']
    delays['time_plus_24'] = delays['time_plus_3']
    
    return delays



def distances_between_hospitals(cleaned_assignments_df: pd.DataFrame):
    """
    Calcuates the Euclidean distance between each hospital using the destination eastings and destination northings.
    Parameters
    ----------
    cleaned_assignments_df : pd.DataFrame
        Fake Assignments data generated from fake data generator
    
    Returns
    -------
    pd.DataFrame of distances between hospitals
    """

    # Creating a data frame consisting of hospital and it's easting and northing
    df_easting_northing = cleaned_assignments_df[['dest_easting', 'dest_northing', 'hospital']].drop_duplicates()
    
    # Adding a dummy column to merge the dataframe to itself, adding _other as a suffix to calculate the Euclidean distance between each hospital
    df_easting_northing['dummy'] = 1
    df_easting_northing = df_easting_northing.merge(df_easting_northing, on='dummy', suffixes=(None, '_other')).drop(columns='dummy')
    df_easting_northing['distance'] = np.sqrt(np.square(df_easting_northing.dest_easting - df_easting_northing.dest_easting_other) 
                                              + np.square(df_easting_northing.dest_northing - df_easting_northing.dest_northing_other))
    
    df_distance = df_easting_northing[['hospital', 'hospital_other', 'distance']]
    
    return df_distance



def rolling_sums_and_means(hospital: str, delays_at_hospital: pd.DataFrame, past_events_all: pd.DataFrame, distances_at_hospital: pd.DataFrame, hours: int):
    """
    Computes rolling aggregations of the past events dataframe aligned to the timestamps of the handover delay.
    Parameters
    ----------
    hospital : str
        A hospital name for which we compute the rolling aggregations.
        
    delays_at_hospital : pd.DataFrane
        Delays dataframe subsetted to the given hospital.
        
    past_events_all : pd.DataFrame
        The past events dataframe for all hospitals.
        
    distances_at_hospital : pd.DataFrame
        The distances from the given hospital to all other hospitals.
    
    
    Returns
    -------
    pd.DataFrame of the situation at the given hospital
    """
    
    # Defining how we want to aggregate the following columns
    sum_mean_dict = {'missing_skill_A':['sum'],
                     'missing_skill_B':['sum'],
                     'missing_skill_C':['sum'],
                     'missing_skill_D':['sum'],
                     'missing_skill_E':['sum'],
                     'num_ambulance_arrivals':['sum'], 
                     'num_ambulance_depatures':['sum'],
                     'flow':['sum'],  
                     'past_delay_mins':['mean'], 
                     'weighted_arrivals_other': ['sum'],
                     '1 to 18': ['sum'],
                     '19 to 36': ['sum'],
                     '37 to 54': ['sum'],
                     '55 to 72': ['sum'],
                     '73 to 90': ['sum'],
                     '90+': ['sum'],
                     '<1': ['sum'],
                     'priority_1':['sum'],
                     'priority_2':['sum'],
                     'priority_3':['sum'],
                     'priority_4':['sum'],
                     'priority_5':['sum'],
                     'priority_6':['sum'],
                     'priority_7':['sum'],
                     'priority_8':['sum'],
                     'priority_9':['sum'],
                     '1_patient_transported':['sum'],
                     '2_patients_transported':['sum'],
                     '3_patients_transported':['sum']
                    }
    
    
    sum_mean_dict = {k: v[0] for k, v in sum_mean_dict.items()}
    
    # Preparing hours we are wanting to add onto our timestamp
    added_hours = datetime.timedelta(hours=hours)
    
    # Dataframe for the past events at the hospital defined in the function 
    local_df = past_events_all[past_events_all['hospital'] == hospital]
    
    # Data frame for the past events of all other hopsitals
    other_hospitals_df = past_events_all[~(past_events_all['hospital'] == hospital)]
    other_hospitals_df = other_hospitals_df.add_suffix('_other')
    
    # Merging other hospitals dataframe to distances
    df_other_hospitals_dist = other_hospitals_df.merge(distances_at_hospital, left_on = 'hospital_other', right_on='hospital', how='left').drop(columns='hospital')
    df_other_hospitals_dist = df_other_hospitals_dist[['hospital_other', 
                                                       'time_other', 
                                                       'num_ambulance_arrivals_other',
                                                       'distance']] 
    
    
    # Adding weighted arrivals column which is an exponentially weighted sum of the number of ambulances arriving at other hospitals depending on the distance to the local hospital
    df_other_hospitals_dist['weighted_arrivals_other'] = df_other_hospitals_dist['num_ambulance_arrivals_other'] * np.exp(-(np.log(2) / 50000) * df_other_hospitals_dist['distance'])
    df_other_hospitals_dist = df_other_hospitals_dist.rename(columns={'hospital_other': 'hospital',
                                                                      'time_other': 'time'})
    
    # Concatenating the other hospital dataframe with local hospital dataframe
    all_hospitals = pd.concat([df_other_hospitals_dist, local_df])
    
    # Adding the added hours input onto the time column
    all_hospitals['time_plus_' + str(hours)] = all_hospitals['time'] + added_hours
    
    # Add column to track which delay the row came from
    delays_at_hospital = delays_at_hospital.rename_axis(index='delay_id').reset_index()

    # Concatenating all hospitals with delays at hospital
    all_hospitals_df = pd.concat([all_hospitals, delays_at_hospital]).set_index(['time_plus_' + str(hours)]).sort_values('time_plus_' + str(hours))

    # Rolling sum, but don't roll the delay ID
    all_hospitals_df_ = all_hospitals_df.drop(columns='delay_id').rolling('6h').agg(sum_mean_dict)
    
    # Add the non-rolled delay ID back
    all_hospitals_df_['delay_id'] = all_hospitals_df['delay_id'].values
    
    delays_at_hospital = delays_at_hospital.sort_values('time_plus_3')
    merged_df = delays_at_hospital.merge(all_hospitals_df_, how='left', on='delay_id')

    return merged_df


def situation_at_all_hospitals(delays: pd.DataFrame, distances: pd.DataFrame, past_events_df: pd.DataFrame, hours: int):
    """
    Performs the rolling sum and mean function to every hospital
    Parameters
    ----------
    delays : pd.DataFrame
        Handover delays at all hospitals.
        
    distances : pd.DataFrame
        Distances to and from all hospitals.
    
    past_events_df : pd.DataFrame
        The past events dataframe at all hospitals.
    
    Returns
    -------
    pd.DataFrame of the situation at all hospitals
    """
    
    dfs = []
    for hospital in HOSPITALS:
        df_delays = delays[delays['hospital'] == hospital]
        df_distances = distances[distances.hospital_other == hospital]
        df_distances = df_distances.drop(columns='hospital_other')
        dfs.append(rolling_sums_and_means(hospital, df_delays, past_events_df, df_distances, hours))

    df = pd.concat(dfs)
    return df


def enrich_data(processed_df: pd.DataFrame, start_year: int):
    """
    Adds open source weather data and bank holiday data to processed dataframe.
    Parameters
    ----------
    processed_df : pd.DataFrame
        The processed dataframe which represents the situation at all hospitals.
    
    start_year : int
        Year at which we start to acquire the weather data
    
    Returns
    -------
    pd.DataFrame of processed data enriched with additional open sourced data
    """
    
    # Adding a time column back in
    processed_df['time'] = processed_df['time_plus_3']
    
    # Adding date, day of week, month, year and hour of day as columns
    processed_df['date'] = processed_df['time'].dt.date.astype('datetime64[ns]')
    processed_df['day_of_week'] = processed_df['time'].dt.day_name()
    processed_df['month'] = processed_df['date'].dt.month
    processed_df['year'] = processed_df['date'].dt.year
    processed_df['hour_of_day'] = processed_df['time'].dt.hour
    
    # Adding bank holiday data
    df_bank_hol = get_bank_holiday_data()
    processed_df = processed_df.merge(df_bank_hol, on=['date'], how='left')
    processed_df.bank_holiday = processed_df.bank_holiday.fillna('No')
    
    # Add weather data
    df_weather = get_weather_data(start_year)
    processed_df = processed_df.merge(df_weather, how='left', on=['year', 'month']) 
    
    return processed_df