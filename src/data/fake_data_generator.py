import pandas as pd
import numpy as np
import datetime
import random

# Specifying lists that we want to randomly pick values from to generate our fake data
hospitals = ['hospital_A',
             'hospital_B',
             'hospital_C',
             'hospital_D',
             'hospital_E']

flow = [1, -1]
num_patients_transported = [1, 2, 3]
age_band = ['<1', '1 to 18', '19 to 36', '37 to 54', '55 to 72', '73 to 90', '90+']
responding_priority = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]


def random_times(start_timestamp: str, end_timestamp: str, number_of_datetime_values: int):
    """
    Randomly generates datetime values.
    Parameters
    ----------
    start_timestamp : str
        Time at which we want to start generating timestamps for.
    end_timestamp : str
        Time at which we want to stop generating timestamps for.
    number_of_datetime_values : int
        Number of rows for which the dataframe should be returned.
    Returns
    -------
    Timestamps
    """
    timestamp_format = '%Y-%m-%d %H:%M:%S'
    start_time = datetime.datetime.strptime(start_timestamp, timestamp_format)
    end_time = datetime.datetime.strptime(end_timestamp, timestamp_format)
    time_diff = end_time - start_time
    return [random.random() * time_diff + start_time for _ in range(number_of_datetime_values)]


def random_crew_skills(number_of_rows: int):
    """
    Randomly generates a dataframe of crew skills.
    Parameters
    ----------
    number_of_rows : int
        Number of rows for which the dataframe should be returned.
    Returns
    -------
    pd.DataFrame of crew skills
    """
    crew_skills = ['skill_A',
                   'skill_B',
                   'skill_C',
                   'skill_D',
                   'skill_E']

    skills = []
    # Randomly generate an integer between 1 and 3 then randomly choose that many values from crew_skills
    for random_integer in np.random.randint(1, 3, size=number_of_rows):
        skills.append(random.choices(crew_skills, k=random_integer))
    skills = pd.DataFrame(skills)
    skills['crew_skills'] = skills[0] + ', ' + skills[1]
    skills = skills[['crew_skills']]
    return skills


def generate_fake_assignments_data(start_timestamp: str, end_timestamp: str, number_of_rows: int):
    """
    Generates fake data for ambulance assignments.
    Parameters
    ----------
    start_timestamp : str
        Date and time at which we want to start generating fake data for.
    end_timestamp : str
        Date and time at which we want to stop generating fake data for.
    number_of_datetime_values : int
        Number of rows for which the dataframe should be returned.
    Returns
    pd.DataFrame for fake assignments data
    """
    # Creating a dataframe where columns are made from randomly choosing times using
    # the random_times function or randomly selecting values from pre-defined lists
    df = pd.DataFrame({'hospital': np.random.choice(hospitals, size=number_of_rows),
                       'num_patients_transported': np.random.choice(num_patients_transported, size=number_of_rows),
                       'time': random_times(start_timestamp, end_timestamp, number_of_rows),
                       'time_destination': random_times(start_timestamp, end_timestamp, number_of_rows),
                       'time_handover': random_times(start_timestamp, end_timestamp, number_of_rows),
                       'time_clear': random_times(start_timestamp, end_timestamp, number_of_rows)})
    # Creating a column for incident number which is a unique identifier
    df = df.rename_axis('incident_number').reset_index()
    # Randomly selecting crew skills
    df['crew1_skill'] = random_crew_skills(number_of_rows)
    df['crew2_skill'] = random_crew_skills(number_of_rows)
    df['crew3_skill'] = random_crew_skills(number_of_rows)
    # Creating a dataframe of eastings and northings where these are random 6 digit integers
    df_easting_northing = pd.DataFrame({'hospital': hospitals,
                                        'dest_easting': np.random.randint(1000000, 2000000, size=len(hospitals)),
                                        'dest_northing': np.random.randint(1000000, 2000000, size=len(hospitals))})
    df = df.merge(df_easting_northing, on='hospital', how='left')
    # When time of handover is greater than time clear, HALOing occurs
    df['haloing_done'] = np.where(df['time_handover'] > df['time_clear'], True, False)
    # When HALOing occurs, calculate the HALOing time in minutes
    df['haloing_time_mins'] = np.where(df['haloing_done'], ((df['time_handover'] - df['time_clear']).
                                                            dt.total_seconds() / 60).round(2), np.nan)
    # Time ambulance left the queue is time clear when HALOing occurs else it is time handover
    df['time_amb_left_queue'] = np.where(df['haloing_done'], df['time_clear'], df['time_handover'])
    # Calculating handover time in minutes
    df['handover_time_mins'] = ((df['time_handover'] - df['time_destination']).dt.total_seconds() / 60).round(2)
    # Handover delay is equal to handover time - 15, when this is less than 0, set value to 0
    df['handover_delay_mins'] = np.where(df['handover_time_mins'] - 15 > 0,
                                         df['handover_time_mins'] - 15,
                                         0)
    df = df.drop(columns=['handover_time_mins'])
    df = df.astype({'time': 'datetime64[s]',
                    'time_destination': 'datetime64[s]',
                    'time_handover': 'datetime64[s]',
                    'time_clear': 'datetime64[s]',
                    'time_amb_left_queue': 'datetime64[s]',
                    'dest_easting': float,
                    'dest_northing': float})
    return df


def generate_fake_pts_data(start_timestamp: str, end_timestamp: str, number_of_rows: int):
    """
    Generates fake data for Patient Transfers System (PTS).
    Parameters
    ----------
    start_timestamp : str
        Date and time at which we want to start generating fake data for.
    end_timestamp : str
        Date and time at which we want to stop generating fake data for.
    number_of_datetime_values : int
        Number of rows for which the dataframe should be returned.
    Returns
    pd.DataFrame for fake PTS data
    """
    # Create a dataframe that randomly choosing from the hospital list, the flow list and randomly generates timestamps
    df = pd.DataFrame({'hospital': np.random.choice(hospitals, size=number_of_rows),
                       'flow': np.random.choice(flow, size=number_of_rows),
                       'time': random_times(start_timestamp, end_timestamp, number_of_rows)})
    df = df.astype({'time': 'datetime64[s]'})
    df = df.set_index(['hospital', 'time'])
    return df


def generate_fake_incidents_data(number_of_rows: int):
    """
    Generates fake data for incidents data.
    Parameters
    ----------
    number_of_datetime_values : int
        Number of rows for which the dataframe should be returned.
    Returns
    pd.DataFrame for fake incidents data
    """
    # Randomly selecting age band values and responding priority values and saving as dataframe
    df = pd.DataFrame({'age_band': np.random.choice(age_band, size=number_of_rows),
                       'responding_priority': np.random.choice(responding_priority, size=number_of_rows)})
    # Creating column for incident number to be used as a unique identifier
    df = df.rename_axis('incident_number').reset_index()
    return df
