#importing Necessary packages
from sklearn.preprocessing import LabelEnconder
import pandas as pd

# IMPORTANT: The data used here has been extracted from a WebCrawler, so it needs to be preprocessed
# to feed into the Deep Learning Regression Model.
# importing data from a .csv file

database = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')


# Preprocessing data


#dropping columns that should't help the regression
database = database.drop('dateCrawled', axis = 1)
database = database.drop('dateCreated', axis = 1)
database = database.drop('nrOfPictures', axis = 1)
database = database.drop('postalCode', axis = 1)
database = database.drop('lastSeen', axis = 1)
database['name'].value_counts()
database = database.drop('name', axis = 1)
database['seller'].value_counts()
database = database.drop('seller', axis = 1)
database['offerType'].value_counts()
database = database.drop('offerType', axis = 1)
 
#locating inconsistent data
inconsistent_price = database.loc[database.price <= 10]
database = database[database.price > 10]

higPrice_vehicles = database.loc[database.price > 350000]
database = database.loc[database.price < 350000]

#fiding Null values
database.loc[pd.isnull(database['vehicleType'])]

#replacing null values with the one that is most frequent in the data base 
database['vehicleType'].value_counts() # most frequent value: 'limousine'

database.loc[pd.isnull(database['gearbox'])]
database['gearbox'].value_counts() #most frequent value: manuell

database.loc[pd.isnull(database['model'])]
database['model'].value_counts() #most frequent value: golf

database.loc[pd.isnull(database['fuelType'])]
database['fuelType'].value_counts() #most frequent value: benzin

database.loc[pd.isnull(database['notRepairedDamage'])]
database['notRepairedDamage'].value_counts() #most frequent value: nein


values = {'vehicleType':'limousine',
          'gearbox':'manuell',
          'model':'golf',
          'fuelType':'benzin',
          'notRepairedDamage':'nein'}


database = database.fillna(value = values)


pred = database.iloc[:, 1:13].values

price = database.iloc[:, 0].values



