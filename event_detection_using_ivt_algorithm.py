import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import matplotlib.ticker as ticker
import seaborn as sns

data_file = pd.read_csv('train.csv',header=None, on_bad_lines='skip', engine='python')

subject_types = ['s2', 's6', 's8', 's14', 's21', 's22', 's28', 's34']

filtered_dataframe = data_file.loc[(data_file[0] == 's2') | ( data_file[0] == 's6') | ( data_file[0] == 's8') | ( data_file[0] == 's14') 
| ( data_file[0] == 's21')| ( data_file[0] == 's22')| ( data_file[0] == 's28') | ( data_file[0] == 's34')]

FIXATION = 'Fixation'
SACCADE = 'Saccade'
THRESHOLD = 400

def create_subjects_array():
  # creating an array for each subject with the coordinates, based on event types: known or unknown.
  result = []
  for s in subject_types:
    subject_data = filtered_dataframe.loc[filtered_dataframe[0] == s]
    subject_dict = {}
    data_false = subject_data.loc[subject_data[1] == False]
    data_true = subject_data.loc[subject_data[1] == True]
    subject_dict['subject_name'] = s
    subject_dict['known_true'] = []
    subject_dict['known_false'] = []
    create_coordinates(data_false, subject_dict['known_false'])
    create_coordinates(data_true, subject_dict['known_true'])
    result.append(subject_dict)
  return result

def create_coordinates(data, subject_dict):
  # take the coordinates from the CSV starting from the second column, two by two
  for index, row in data.iterrows():
      row = row[2:]
      subject_dict.extend([*zip(row[::2], row[1::2])])

subjects = create_subjects_array()

def compute_distances(gaze_points):
  # Calculate velocities for each point: euclidian distance between two consecutive points
  return [np.linalg.norm(p_i - p_i_1) for p_i_1, p_i in gaze_points]


def compute_velocities(gaze_points, threshold):
  # Calculate velocities for each point: euclidian distance between two consecutive points
  distances = compute_distances(gaze_points)
  events = []
  index = 0
  mean_dist = np.mean([distance for distance in distances if not math.isnan(distance)])
  print(f"MIN DIST = {mean_dist}")
  for distance in distances:
    # comparing the distance with the velocity threshold
    # in case the distance is lower than the threshold, the corresponding coordinates of that distance are the fixation points
    # otherwise, those are saccade points
    # each pair of coordinates has the event type: FIXATION or SACCADE
    if distance < threshold:
      events.append([FIXATION, gaze_points[index][0], gaze_points[index][1]])
    else:
      events.append([SACCADE, gaze_points[index][0], gaze_points[index][1]])
    index += 1
  return events

def compute_fixations_and_saccades(events):
  group_fixations = []
  saccades = []
  fixations = []
  # loopting through the events marked while computing the velocities
  for event in events:
    centroid_x = 0
    centroid_y = 0
    if math.isnan(event[1]) or math.isnan(event[2]):
      continue
    if event[0] == FIXATION:
      # each event that is a fixation is put into a group
      # the group will have the collection of consecutive fixations
      group_fixations.append((event[1], event[2]))
    else: 
      saccades.append((event[1], event[2]))
      # when a saccade was found, then we have also found one group of fixations
      # the group of fixations is reduced to one pair of coordinates, which is the centroid point of those fixations
      size_of_group_fixations = len(group_fixations)
      if size_of_group_fixations > 1:
        # we found a group so we compute the centroid
        # we add the pair of coordinates for the centroid calculated above to the fixations array
        centroid_x = sum([fixation[0] for fixation in group_fixations]) / size_of_group_fixations
        centroid_y = sum([fixation[1] for fixation in group_fixations]) / size_of_group_fixations
        fixations.append((centroid_x, centroid_y))
      if size_of_group_fixations == 1:
        # in case there is no centroid computed, then in the group of fixations, there is only one single element
        # that element is added to the fixations array
        fixations.append(group_fixations[0])
      group_fixations = []

  return fixations, saccades

def get_events_per_subject(subjects):
  """Compute mean fixation duration and mean saccades amplitude for both known and unknown events for each subject.
  
  Compute mean fixtion duration and mean saccadeds amplitude events for each subjects."""
  result = {}

  for each in subjects: 
    # for each subject type, we're computing the fixations and saccades points based on the algorithm of 
    # calculating the velocities. 
    subject_name = each['subject_name']
    events_true = compute_velocities(each['known_true'], THRESHOLD)
    fixations_true, saccades_true = compute_fixations_and_saccades(events_true)
    print(f"{len(events_true)} true events in total for subject:{subject_name}")
    print(f"{len(fixations_true)} fixations for known events for subject:{subject_name}")
    print(f"{len(saccades_true)} saccades for known events for subject: {subject_name}")

    events_false = compute_velocities(each['known_false'], THRESHOLD)
    fixations_false, saccades_false = compute_fixations_and_saccades(events_false)
    print(f"{len(events_false)} false events in total for subject:{subject_name}")
    print(f"{ len(fixations_false)} fixations for unknown events for subject: {subject_name}")
    print(f"{len(saccades_false)} saccades for unknown events for subject: {subject_name}")

    all_fixations_per_subject = fixations_true + fixations_false
    all_saccades_per_subject = saccades_true + saccades_false

    result[subject_name] = {
        "mfd_true": compute_mean(fixations_true),
        "mfd_false": compute_mean(fixations_false),
        "msa_true": compute_mean(saccades_true),
        "msa_false": compute_mean(saccades_false),
        "mfd_sd_true": compute_standard_deviation(fixations_true),
        "msa_sd_true": compute_standard_deviation(saccades_true),
        "mfd_sd_false": compute_standard_deviation(fixations_false),
        "msa_sd_false": compute_standard_deviation(saccades_false)
    }
    result[subject_name]["mfd_overall"] = compute_mean(all_fixations_per_subject)
    result[subject_name]["mfd_overall_sd"] = compute_standard_deviation(all_fixations_per_subject)
    result[subject_name]["msa_overall"] = compute_mean(all_saccades_per_subject)
    result[subject_name]["msa_overall_sd"] = compute_standard_deviation(all_saccades_per_subject)

  
  return result
  

  
def compute_mean(array):
  """Compute mean"""
  return np.mean(array)

def compute_standard_deviation(array):
  """Compute standard deviation"""
  return np.std(array)

means = get_events_per_subject(subjects)
print(means)
def write_results_to_csv(result):
  """Create a CSV file for the results."""
  with open("results.csv", "w", newline='') as file_name:
    writer = csv.writer(file_name)
    headers = ["subject_id", "MFD_true", "MFD_SD_true", "MFD_false", "MFD_SD_false", "MSA_true", "MSA_SD_true", "MSA_false", "MSA_SD_false", "MFD_overall", "MFD_overall_SD", "MSA_overall", "MSA_overall_SD"]
    writer.writerow(headers)
    for subject_name in subject_types:
      row = []
      subject_data = result[subject_name]
      row.append(subject_name)
      row.append(subject_data["mfd_true"])
      row.append(subject_data["mfd_sd_true"])
      row.append(subject_data["mfd_false"])
      row.append(subject_data["mfd_sd_false"])
      row.append(subject_data["msa_true"])
      row.append(subject_data["msa_sd_true"])
      row.append(subject_data["msa_false"])
      row.append(subject_data["msa_sd_false"])
      row.append(subject_data["mfd_overall"])
      row.append(subject_data["mfd_overall_sd"])
      row.append(subject_data["msa_overall"])
      row.append(subject_data["msa_overall_sd"])
      
      writer.writerow(row)


write_results_to_csv(means)

result_df = pd.read_csv('results.csv')
    
plt.rcParams["figure.figsize"] = (25,15)
ax = result_df.plot.bar()
plt.xticks(range(len(result_df)), result_df["subject_id"])
plt.xlabel("Subjects ID")
plt.ylabel("Value")
plt.legend()
plt.title('Eye Tracking Data')
plt.show()